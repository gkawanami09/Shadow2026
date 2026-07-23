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

    def servo(self, nome, angulo):
        """Move um servo do PCA9685 sem alterar o keepalive dos motores."""
        nome = str(nome).upper()
        if nome not in ("GARRA_ESQ", "GARRA_DIR", "CACAMBA", "FUTABA"):
            raise ValueError(f"Servo invalido: {nome}")
        angulo = int(round(angulo))
        if not 0 <= angulo <= 180:
            raise ValueError(f"Angulo fora de 0..180: {angulo}")
        self._send_aux_cmd(f"SERVO {nome} {angulo}")

    def led(self, modo):
        """Define o LED como APAGADO ou ACESO."""
        modo = str(modo).upper()
        if modo not in ("APAGADO", "ACESO"):
            raise ValueError(f"Modo de LED invalido: {modo}")
        self._send_aux_cmd(f"LED {modo}")

    def distancia_ultrassom(self, timeout=0.2):
        """Solicita uma leitura e retorna a distancia em mm, ou None sem eco."""
        resposta = self._query("ULTRASSOM", "OK ULTRASSOM ", timeout)
        if resposta is None:
            return None
        try:
            distancia_mm = int(resposta.split()[-1])
        except (ValueError, IndexError):
            return None
        return None if distancia_mm < 0 else distancia_mm

    def refresh(self):
        """Re-send the last command if the keepalive interval elapsed
        (call inside any sleep while motors are running)."""
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
        self._drain()
        self._write_line(cmd)
        deadline = time.monotonic() + timeout
        while self._connected and time.monotonic() < deadline:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="replace").strip()
                    if line.startswith(prefix):
                        return line
                    if line.startswith("ERRO"):
                        print(f"[serial] firmware respondeu: {line}")
                else:
                    time.sleep(0.002)
            except (serial.SerialException, OSError):
                self._connected = False
                break
        return None

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
            while self._ser.in_waiting:
                line = self._ser.readline().decode(errors="replace").strip()
                if line.startswith("ERRO"):
                    print(f"[serial] firmware respondeu: {line}")
        except (serial.SerialException, OSError):
            self._connected = False

    def _try_reconnect(self):
        now = time.monotonic()
        if now - self._last_reconnect_t < config.SERIAL_RECONNECT_BACKOFF:
            return
        self._last_reconnect_t = now
        for device in self._candidate_ports():
            if self._try_port(device):
                print(f"[serial] reconectado em {device}")
                return
