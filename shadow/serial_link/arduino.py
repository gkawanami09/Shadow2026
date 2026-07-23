"""
serial_link/arduino.py — USB-serial link to the Shadow2026 Arduino Uno (SPEC 01).

There is nothing to port from OE² here (the OE² Pi drove its L298N straight
from GPIO — dossier Section 4). This class speaks the protocol ALREADY flashed
on the Shadow2026 Uno (../Shadow2026/arduino/motor_controller/, "SPEC 01"),
documented in serial_link/PROTOCOL.md.

Responsibilities (mission §3.1, adapted to the existing firmware):
  - auto-detect the port (/dev/ttyACM* then /dev/ttyUSB*; COM* as bench fallback),
    handshake via boot banner "Arduino pronto - SPEC 01" or PING→PONG,
    retry with 0.5 s backoff, give up after 5 s total;
  - never block on reads after boot — replies (OK …/ERRO …) are drained
    non-blockingly; ERRO lines are printed;
  - clamp + assert |pwm| <= MAX_PWM (120) on every outgoing speed;
  - dedupe identical commands (min 50 ms apart) so busy-wait loops calling
    steer() every 1 ms don't flood the 115200 link;
  - refresh(): re-send the last command every 0.25 s so the Uno's 1 s watchdog
    never fires mid-maneuver (OE² sleeps up to 1.35 s with motors running);
  - on serial errors: close, reopen with backoff; while disconnected commands
    are dropped (the Uno watchdog keeps the motors stopped).
"""

import time

import serial
from serial.tools import list_ports

import config


class Arduino:
    def __init__(self, port=None):
        self._ser = None
        self._last_cmd = None
        self._last_send_t = 0.0
        self._last_reconnect_t = 0.0
        self._connected = False
        self._desired_led_mode = None
        self._rx_buffer = bytearray()
        self._ultra_pending = False
        self._ultra_deadline = 0.0
        self._ultra_ready = False
        self._ultra_value = None
        self._manual_pending = False
        self._manual_response = None

        if port is not None:
            if not self._try_port(port):
                raise RuntimeError(f"Arduino nao respondeu na porta {port}")
        else:
            self._autodetect()

    # ------------------------------------------------------------------ setup

    def _candidate_ports(self):
        devices = [p.device for p in list_ports.comports()]
        ordered = []
        for prefix in config.SERIAL_PORT_PREFIXES:
            ordered += sorted(d for d in devices if d.startswith(prefix) and d not in ordered)
        return ordered

    def _try_port(self, device):
        """Open a port and handshake: wait for the boot banner (the Uno
        auto-resets when the port opens), else probe with PING."""
        try:
            ser = serial.Serial(device, config.SERIAL_BAUD, timeout=0.2)
        except (serial.SerialException, OSError):
            return False

        deadline = time.monotonic() + 2.5
        try:
            while time.monotonic() < deadline:
                line = ser.readline().decode(errors="replace").strip()
                if config.SERIAL_BANNER in line:
                    print(f"[serial] {device}: '{line}'")
                    self._adopt(ser)
                    return True

            # Sem banner (placa ja estava ligada e sem auto-reset): tenta PING.
            for _ in range(3):
                ser.reset_input_buffer()
                ser.write(b"PING\n")
                reply_deadline = time.monotonic() + 0.4
                while time.monotonic() < reply_deadline:
                    line = ser.readline().decode(errors="replace").strip()
                    if line == "PONG":
                        print(f"[serial] {device}: PONG")
                        self._adopt(ser)
                        return True
        except (serial.SerialException, OSError):
            pass

        try:
            ser.close()
        except (serial.SerialException, OSError):
            pass
        return False

    def _adopt(self, ser):
        ser.timeout = 0  # nunca mais bloquear em leitura
        self._ser = ser
        self._connected = True
        self._rx_buffer.clear()
        self.cancelar_ultrassom()
        self._manual_pending = False
        self._manual_response = None

    def _autodetect(self):
        deadline = time.monotonic() + config.SERIAL_HANDSHAKE_TIMEOUT
        while time.monotonic() < deadline:
            for device in self._candidate_ports():
                if self._try_port(device):
                    return
            time.sleep(config.SERIAL_RETRY_BACKOFF)
        raise RuntimeError(
            "Arduino nao encontrado. Verifique o cabo USB e se o firmware "
            "SPEC 01 esta gravado (banner 'Arduino pronto - SPEC 01')."
        )

    # ------------------------------------------------------------- public API

    def lado(self, esq, dir_):
        """LADO <esq> <dir> — signed wheel speeds, left pair / right pair."""
        esq, dir_ = int(round(esq)), int(round(dir_))
        assert abs(esq) <= config.MAX_PWM and abs(dir_) <= config.MAX_PWM, \
            f"PWM acima do teto de seguranca ({esq}, {dir_}) > {config.MAX_PWM}"
        self._send_cmd(f"LADO {esq} {dir_}")

    def rodas(self, fe, te, fd, td):
        """RODAS <FE> <TE> <FD> <TD> — PWM assinado por motor."""
        velocidades = tuple(int(round(v)) for v in (fe, te, fd, td))
        assert all(abs(v) <= config.MAX_PWM for v in velocidades), \
            f"PWM acima do teto de seguranca {velocidades} > {config.MAX_PWM}"
        self._send_cmd("RODAS " + " ".join(map(str, velocidades)))

    def parar(self):
        self._send_cmd("PARAR")

    def ping(self):
        self._send_cmd("PING", force=True)

    def servo(self, nome, deslocamento):
        """Move o servo relativamente a ultima posicao comandada, em graus."""
        nome = str(nome).upper()
        if nome == "FUTABA":
            raise ValueError("Servo FUTABA esta desativado no firmware")
        if nome not in ("GARRA_ESQ", "GARRA_DIR", "CACAMBA"):
            raise ValueError(f"Servo invalido: {nome}")
        deslocamento = int(round(deslocamento))
        if not -180 <= deslocamento <= 180:
            raise ValueError(f"Deslocamento fora de -180..180: {deslocamento}")
        self._send_aux_cmd(f"SERVO {nome} {deslocamento}")

    def led(self, modo):
        """Define o LED como APAGADO ou ACESO."""
        modo = str(modo).upper()
        if modo not in ("APAGADO", "ACESO"):
            raise ValueError(f"Modo de LED invalido: {modo}")
        # O Uno reinicia com o LED aceso ao reabrir a USB. Guardar o modo
        # desejado permite restaura-lo automaticamente numa reconexao.
        self._desired_led_mode = modo
        self._send_aux_cmd(f"LED {modo}")

    def distancia_ultrassom(self, timeout=0.2):
        """Solicita uma leitura e retorna a distancia em mm, ou None sem eco."""
        if not self.iniciar_ultrassom(timeout=timeout):
            return None
        while True:
            concluido, distancia_mm = self.poll_ultrassom()
            if concluido:
                return distancia_mm
            time.sleep(0.002)

    def iniciar_ultrassom(self, timeout=0.2):
        """Inicia uma leitura sem esperar a resposta do firmware."""
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout do ultrassom deve ser positivo")
        self._drain()
        if (
            not self._connected
            or self._ultra_pending
            or self._ultra_ready
        ):
            return False

        self._ultra_pending = True
        self._ultra_deadline = time.monotonic() + timeout
        self._ultra_value = None
        self._write_line("ULTRASSOM")
        if not self._connected:
            self._ultra_pending = False
            return False
        return True

    def poll_ultrassom(self):
        """Retorna (concluido, distancia_mm) sem bloquear."""
        self._drain()
        now = time.monotonic()
        if self._ultra_pending and (
            not self._connected or now >= self._ultra_deadline
        ):
            self._ultra_pending = False
            self._ultra_ready = True
            self._ultra_value = None

        if not self._ultra_ready:
            return False, None
        value = self._ultra_value
        self._ultra_ready = False
        self._ultra_value = None
        return True, value

    def cancelar_ultrassom(self):
        """Descarta pedido/resposta; uma resposta tardia nao sera reutilizada."""
        self._ultra_pending = False
        self._ultra_ready = False
        self._ultra_value = None
        self._ultra_deadline = 0.0

    def futaba(self, potencia, tempo_ms):
        """Aciona o servo continuo com potencia -100..100 por ate 3000 ms."""
        potencia = int(round(potencia))
        tempo_ms = int(round(tempo_ms))
        if potencia == 0 or not -100 <= potencia <= 100:
            raise ValueError("Potencia do Futaba deve estar em -100..-1 ou 1..100")
        if not 1 <= tempo_ms <= 3000:
            raise ValueError("Tempo do Futaba deve estar em 1..3000 ms")
        self._send_aux_cmd(f"FUTABA {potencia} {tempo_ms}")

    def parar_futaba(self):
        """Corta imediatamente o sinal do canal continuo CH3."""
        self._send_aux_cmd("FUTABA PARAR")

    def comando_serial(self, comando, timeout=0.5):
        """Envia uma linha livre e retorna a primeira resposta do firmware.

        Destinado a ferramentas manuais de teste. Nao substitui o ultimo
        comando de movimento usado pelo keepalive da aplicacao principal.
        """
        comando = str(comando).strip()
        if not comando:
            raise ValueError("O comando serial nao pode estar vazio")

        if not self._connected:
            self._try_reconnect()
        if not self._connected:
            return None

        self._drain()
        self._manual_pending = True
        self._manual_response = None
        self._write_line(comando)
        deadline = time.monotonic() + timeout
        try:
            while self._connected and time.monotonic() < deadline:
                self._drain()
                if self._manual_response is not None:
                    return self._manual_response
                time.sleep(0.002)
            return None
        finally:
            self._manual_pending = False
            self._manual_response = None

    def refresh(self, fail_closed=False):
        """Re-send the last command if the keepalive interval elapsed
        (call inside any sleep while motors are running).

        ``fail_closed`` e usado pelo resgate: depois de reconectar, substitui
        um movimento baseado numa imagem antiga por PARAR. O padrao preserva
        o comportamento dos outros modos existentes.
        """
        if fail_closed and not self._connected:
            self._try_reconnect()
            if not self._connected:
                return
            # Uma reconexao pode acontecer muito depois do frame que gerou o
            # ultimo movimento. Nunca ressuscitar esse comando antigo.
            self._write_line("PARAR")
            self._last_cmd = "PARAR"
            self._last_send_t = time.monotonic()
            self._drain()
            return
        if self._last_cmd is not None and \
                time.monotonic() - self._last_send_t > config.SERIAL_KEEPALIVE_S:
            self._write_line(self._last_cmd)
            self._last_send_t = time.monotonic()
        self._drain()

    def close(self):
        if self._ser is not None:
            try:
                self._write_line("PARAR")
                time.sleep(0.05)
                self._ser.close()
            except (serial.SerialException, OSError):
                pass
        self._connected = False
        self._ser = None

    # --------------------------------------------------------------- plumbing

    def _send_cmd(self, cmd, force=False):
        now = time.monotonic()
        if not force and cmd == self._last_cmd and \
                now - self._last_send_t < config.SERIAL_MIN_RESEND_S:
            return
        self._write_line(cmd)
        self._last_cmd = cmd
        self._last_send_t = now
        self._drain()

    def _send_aux_cmd(self, cmd):
        """Envia periferico sem substituir o ultimo comando dos motores."""
        self._write_line(cmd)
        self._drain()

    def _query(self, cmd, prefix, timeout):
        """Envia uma consulta e aguarda somente a resposta correspondente."""
        line = self.comando_serial(cmd, timeout=timeout)
        return line if line is not None and line.startswith(prefix) else None

    def _write_line(self, cmd):
        if not self._connected:
            self._try_reconnect()
            if not self._connected:
                return
        try:
            self._ser.write((cmd + "\n").encode())
        except (serial.SerialException, OSError) as err:
            print(f"[serial] erro de escrita ({err}); reconectando…")
            self._connected = False
            try:
                self._ser.close()
            except (serial.SerialException, OSError):
                pass

    def _drain(self):
        """Non-blocking read of pending replies; surfaces ERRO lines."""
        if not self._connected:
            return
        try:
            available = int(self._ser.in_waiting)
            if available <= 0:
                return
            chunk = self._ser.read(available)
            if isinstance(chunk, str):
                chunk = chunk.encode()
            self._rx_buffer.extend(chunk)

            while b"\n" in self._rx_buffer:
                raw_line, _, remainder = self._rx_buffer.partition(b"\n")
                self._rx_buffer = bytearray(remainder)
                line = raw_line.decode(errors="replace").strip()
                if line:
                    self._route_line(line)

            # Firmware correto sempre termina linhas com LF. Limitar lixo de
            # uma porta incorreta sem perder fragmentos normais.
            if len(self._rx_buffer) > 4096:
                self._rx_buffer.clear()
        except (serial.SerialException, OSError):
            self._connected = False

    def _route_line(self, line):
        if line.startswith("OK ULTRASSOM "):
            if self._ultra_pending:
                try:
                    value = int(line.split()[-1])
                except (ValueError, IndexError):
                    value = -1
                self._ultra_value = None if value < 0 else value
                self._ultra_pending = False
                self._ultra_ready = True
                return
            # A ferramenta manual tambem pode enviar ULTRASSOM.
            if self._manual_pending and self._manual_response is None:
                self._manual_response = line
            return
        if self._manual_pending and self._manual_response is None:
            self._manual_response = line
        if line.startswith("ERRO"):
            print(f"[serial] firmware respondeu: {line}")

    def _try_reconnect(self):
        now = time.monotonic()
        if now - self._last_reconnect_t < config.SERIAL_RECONNECT_BACKOFF:
            return
        self._last_reconnect_t = now
        for device in self._candidate_ports():
            if self._try_port(device):
                print(f"[serial] reconectado em {device}")
                if self._desired_led_mode is not None:
                    self._send_aux_cmd(
                        f"LED {self._desired_led_mode}")
                return
