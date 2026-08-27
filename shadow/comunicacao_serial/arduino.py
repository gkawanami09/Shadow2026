"""Mantém a comunicação USB serial com o Arduino usando o protocolo SPEC 01."""

from collections import deque
from dataclasses import dataclass
import time

import serial
from serial.tools import list_ports

import config


@dataclass(frozen=True)
class LeituraMpu:
    """Amostra de orientacao devolvida pelo comando ``MPU`` do Uno.

    O yaw e relativo: o MPU6050 nao possui magnetometro. Controles longos
    devem usar ``MPU ZERO``; manobras curtas podem usar a diferenca entre a
    amostra inicial e a atual, sem depender do zero acumulado.
    """

    pitch_graus: float
    roll_graus: float
    yaw_graus: float


class Arduino:
    def __init__(self, port=None):
        self._ser = None
        self._last_cmd = None
        self._last_send_t = 0.0
        self._last_reconnect_t = 0.0
        self._connected = False
        self._connection_epoch = 0
        self._desired_led_mode = None
        self._rx_buffer = bytearray()
        self._ultra_pending = False
        self._ultra_deadline = 0.0
        self._ultra_ready = False
        self._ultra_value = None
        self._ultra_response_received = False
        self._ultra_lado = "FRENTE"
        self._mpu_pending = False
        self._mpu_deadline = 0.0
        self._mpu_ready = False
        self._mpu_value = None
        self._rampa_pending = False
        self._rampa_deadline = 0.0
        self._rampa_ready = False
        self._rampa_estado = None
        self._rampa_angulo = None
        self._manual_pending = False
        self._manual_responses = deque()
        self._reconexao_automatica = True

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
        self._connection_epoch += 1
        self._rx_buffer.clear()
        self.cancelar_ultrassom()
        self.cancelar_mpu()
        self.cancelar_rampa()
        self._manual_pending = False
        self._manual_responses.clear()

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

    @property
    def connection_epoch(self):
        """Muda sempre que uma nova conexao serial e adotada."""
        return self._connection_epoch

    @property
    def connected(self):
        return self._connected

    def travar_sessao(self):
        """Desliga reconexao automatica ate esta instancia ser descartada."""
        self._reconexao_automatica = False

    @property
    def ultima_leitura_ultrassom_respondeu(self):
        """Distingue ``sem eco`` do firmware de timeout da comunicacao.

        ``poll_ultrassom()`` continua com o contrato antigo: ambos devolvem
        distancia ``None``. Este sinal adicional permite que rotinas de
        seguranca bloqueiem uma manobra quando o Arduino nem chegou a
        responder, sem confundir isso com um ambiente realmente sem eco.
        """
        return bool(self._ultra_response_received)

    @property
    def consultas_sensores_pendentes(self):
        """Indica se ha consulta de ultrassom ou MPU aguardando resposta."""
        return bool(self._ultra_pending or self._mpu_pending)

    def lado(self, esq, dir_):
        """LADO <esq> <dir> — signed wheel speeds, left pair / right pair."""
        esq, dir_ = int(round(esq)), int(round(dir_))
        assert abs(esq) <= config.MAX_PWM and abs(dir_) <= config.MAX_PWM, \
            f"PWM acima do teto de seguranca ({esq}, {dir_}) > {config.MAX_PWM}"
        return self._send_cmd(f"LADO {esq} {dir_}")

    def rodas(self, fe, te, fd, td):
        """RODAS <FE> <TE> <FD> <TD> — PWM assinado por motor."""
        velocidades = tuple(int(round(v)) for v in (fe, te, fd, td))
        assert all(abs(v) <= config.MAX_PWM for v in velocidades), \
            f"PWM acima do teto de seguranca {velocidades} > {config.MAX_PWM}"
        return self._send_cmd(
            "RODAS " + " ".join(map(str, velocidades)))

    def parar(self):
        return self._send_cmd("PARAR")

    def ping(self):
        return self._send_cmd("PING", force=True)

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
        return self._send_aux_cmd(f"SERVO {nome} {deslocamento}")

    def garras(self, deslocamento_esq, deslocamento_dir):
        """Envia os dois deltas no mesmo pacote USB.

        O Uno aplica CH0 e CH1 sequencialmente, mas uma unica escrita impede
        que outro comando seja intercalado e reduz a diferenca a poucos ms.
        """
        deslocamento_esq = int(round(deslocamento_esq))
        deslocamento_dir = int(round(deslocamento_dir))
        for deslocamento in (deslocamento_esq, deslocamento_dir):
            if not -180 <= deslocamento <= 180:
                raise ValueError(
                    f"Deslocamento fora de -180..180: {deslocamento}")
        return self._send_aux_batch((
            f"SERVO GARRA_ESQ {deslocamento_esq}",
            f"SERVO GARRA_DIR {deslocamento_dir}",
        ))

    def led(self, modo):
        """Define o LED como APAGADO ou ACESO."""
        modo = str(modo).upper()
        if modo not in ("APAGADO", "ACESO"):
            raise ValueError(f"Modo de LED invalido: {modo}")
        # O Uno reinicia com o LED aceso ao reabrir a USB. Guardar o modo
        # desejado permite restaura-lo automaticamente numa reconexao.
        self._desired_led_mode = modo
        return self._send_aux_cmd(f"LED {modo}")

    def distancia_ultrassom(self, timeout=0.2, lado="FRENTE"):
        """Solicita uma leitura e retorna a distancia em mm, ou None sem eco."""
        if not self.iniciar_ultrassom(timeout=timeout, lado=lado):
            return None
        while True:
            concluido, distancia_mm = self.poll_ultrassom()
            if concluido:
                return distancia_mm
            time.sleep(0.002)

    def iniciar_ultrassom(self, timeout=0.2, lado="FRENTE"):
        """Inicia uma leitura sem esperar a resposta do firmware."""
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout do ultrassom deve ser positivo")
        lado = str(lado).upper()
        if lado not in ("FRENTE", "LATERAL"):
            raise ValueError("lado do ultrassom deve ser FRENTE ou LATERAL")
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
        self._ultra_response_received = False
        self._ultra_lado = lado
        # Sempre explicita o lado. O terminal manual validado no robo usa
        # ``ULTRASSOM FRENTE``; depender da abreviacao legada sem argumento
        # pode ler outro sensor em um firmware antigo ainda gravado no Uno.
        comando = f"ULTRASSOM {lado}"
        self._write_line(comando)
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
            self._ultra_response_received = False

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
        self._ultra_response_received = False
        self._ultra_deadline = 0.0
        self._ultra_lado = "FRENTE"

    def iniciar_mpu(self, timeout=0.12):
        """Pede uma amostra do MPU sem bloquear o ciclo de motores."""
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout do MPU deve ser positivo")
        self._drain()
        if (
            not self._connected
            or self._mpu_pending
            or self._mpu_ready
        ):
            return False
        self._mpu_pending = True
        self._mpu_deadline = time.monotonic() + timeout
        self._mpu_value = None
        self._write_line("MPU")
        if not self._connected:
            self._mpu_pending = False
            return False
        return True

    def poll_mpu(self):
        """Retorna ``(concluido, LeituraMpu | None)`` sem bloquear."""
        self._drain()
        now = time.monotonic()
        if self._mpu_pending and (
            not self._connected or now >= self._mpu_deadline
        ):
            self._mpu_pending = False
            self._mpu_ready = True
            self._mpu_value = None
        if not self._mpu_ready:
            return False, None
        leitura = self._mpu_value
        self._mpu_ready = False
        self._mpu_value = None
        return True, leitura

    def cancelar_mpu(self):
        """Descarta uma leitura pendente do MPU."""
        self._mpu_pending = False
        self._mpu_ready = False
        self._mpu_value = None
        self._mpu_deadline = 0.0

    def zerar_mpu(self, timeout=0.5):
        """Zera a referencia relativa de pitch, roll e yaw com o robo parado."""
        # Uma amostra assincrona do MPU pedida pela fase anterior pode chegar
        # exatamente depois deste comando. Ela nao e a confirmacao do ZERO e
        # jamais deve impedir que a resposta ``OK MPU ZERO`` seja recebida.
        self.cancelar_mpu()
        resposta = self._query("MPU ZERO", "OK MPU ZERO", timeout)
        return resposta is not None

    def iniciar_rampa(self, timeout=None):
        """Solicita o estado de rampa sem bloquear o segue-linha.

        O firmware responde ``OK RAMPA ESTADO=<...> ANGULO=<...>``. A leitura
        e separada dos comandos de motor para que o keepalive continue sendo o
        ultimo movimento enviado.
        """
        if timeout is None:
            timeout = config.RAMPA_RESPOSTA_TIMEOUT_S
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout da rampa deve ser positivo")
        self._drain()
        if (
            not self._connected
            or self._rampa_pending
            or self._rampa_ready
        ):
            return False

        self._rampa_pending = True
        self._rampa_deadline = time.monotonic() + timeout
        self._rampa_estado = None
        self._rampa_angulo = None
        self._write_line("RAMPA")
        if not self._connected:
            self._rampa_pending = False
            return False
        return True

    def poll_rampa(self):
        """Retorna ``(concluido, (estado, angulo))`` sem esperar a serial.

        Quando nao houver resposta valida, retorna ``(True, None)``. O
        controle trata isso como plano, preservando a velocidade normal.
        """
        self._drain()
        now = time.monotonic()
        if self._rampa_pending and (
            not self._connected or now >= self._rampa_deadline
        ):
            self._rampa_pending = False
            self._rampa_ready = True
            self._rampa_estado = None
            self._rampa_angulo = None

        if not self._rampa_ready:
            return False, None
        estado, angulo = self._rampa_estado, self._rampa_angulo
        self._rampa_ready = False
        self._rampa_estado = None
        self._rampa_angulo = None
        return True, None if estado is None else (estado, angulo)

    def cancelar_rampa(self):
        """Descarta a consulta de inclinacao pendente ou ja recebida."""
        self._rampa_pending = False
        self._rampa_ready = False
        self._rampa_estado = None
        self._rampa_angulo = None
        self._rampa_deadline = 0.0

    def futaba(self, potencia, tempo_ms):
        """Aciona o servo continuo com potencia -100..100 por ate 3000 ms."""
        potencia = int(round(potencia))
        tempo_ms = int(round(tempo_ms))
        if potencia == 0 or not -100 <= potencia <= 100:
            raise ValueError("Potencia do Futaba deve estar em -100..-1 ou 1..100")
        if not 1 <= tempo_ms <= 3000:
            raise ValueError("Tempo do Futaba deve estar em 1..3000 ms")
        return self._send_aux_cmd(f"FUTABA {potencia} {tempo_ms}")

    def parar_futaba(self):
        """Corta imediatamente o sinal do canal continuo CH3."""
        return self._send_aux_cmd("FUTABA PARAR")

    def comando_serial(self, comando, timeout=0.5, resposta_esperada=None):
        """Envia uma linha livre e retorna a primeira resposta do firmware.

        Destinado a ferramentas manuais de teste. Nao substitui o ultimo
        comando de movimento usado pelo keepalive da aplicacao principal.

        Quando ``resposta_esperada`` e informado, respostas pendentes de
        comandos anteriores sao descartadas ate a resposta com esse prefixo
        chegar. Isso e necessario para comandos de seguranca, como MPU ZERO,
        que podem disputar o buffer com uma leitura assincrona anterior.
        """
        comando = str(comando).strip()
        if not comando:
            raise ValueError("O comando serial nao pode estar vazio")

        if not self._connected and getattr(
                self, "_reconexao_automatica", True):
            self._try_reconnect()
        if not self._connected:
            return None

        self._drain()
        self._manual_pending = True
        self._manual_responses.clear()
        self._write_line(comando)
        deadline = time.monotonic() + timeout
        try:
            while self._connected and time.monotonic() < deadline:
                self._drain()
                while self._manual_responses:
                    resposta = self._manual_responses.popleft()
                    if (
                        resposta_esperada is None
                        or resposta.startswith(resposta_esperada)
                    ):
                        return resposta
                time.sleep(0.002)
            return None
        finally:
            self._manual_pending = False
            self._manual_responses.clear()

    def refresh(self, fail_closed=False):
        """Re-send the last command if the keepalive interval elapsed
        (call inside any sleep while motors are running).

        ``fail_closed`` e usado pelo resgate: depois de reconectar, substitui
        um movimento baseado numa imagem antiga por PARAR. O padrao preserva
        o comportamento dos outros modos existentes.
        """
        if fail_closed and not self._connected:
            if getattr(self, "_reconexao_automatica", True):
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
            return True
        sent = self._write_line(cmd)
        self._last_cmd = cmd
        self._last_send_t = now
        self._drain()
        return sent

    def _send_aux_cmd(self, cmd):
        """Envia periferico sem substituir o ultimo comando dos motores."""
        sent = self._write_line(cmd)
        self._drain()
        return sent

    def _send_aux_batch(self, commands):
        """Envia varias linhas validadas em uma unica escrita serial."""
        commands = tuple(str(command).strip() for command in commands)
        if (
            not commands
            or any(
                not command
                or "\n" in command
                or "\r" in command
                for command in commands
            )
        ):
            raise ValueError("Lote serial auxiliar invalido")
        sent = self._write_line("\n".join(commands))
        self._drain()
        return sent

    def _query(self, cmd, prefix, timeout):
        """Envia uma consulta e aguarda somente a resposta correspondente."""
        return self.comando_serial(
            cmd,
            timeout=timeout,
            resposta_esperada=prefix,
        )

    def _write_line(self, cmd):
        if not self._connected:
            if getattr(self, "_reconexao_automatica", True):
                self._try_reconnect()
            if not self._connected:
                return False
        try:
            self._ser.write((cmd + "\n").encode())
            return True
        except (serial.SerialException, OSError) as err:
            print(f"[serial] erro de escrita ({err}); reconectando…")
            self._connected = False
            try:
                self._ser.close()
            except (serial.SerialException, OSError):
                pass
            return False

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
        except (serial.SerialException, OSError) as err:
            print(f"[serial] erro de leitura ({err}); conexao encerrada")
            self._connected = False
            if self._ser is not None:
                try:
                    self._ser.close()
                except (serial.SerialException, OSError):
                    pass

    def _route_line(self, line):
        if line.startswith("OK MPU "):
            if self._mpu_pending:
                campos = dict(
                    campo.split("=", 1)
                    for campo in line.split()[2:]
                    if "=" in campo
                )
                try:
                    self._mpu_value = LeituraMpu(
                        pitch_graus=float(campos["PITCH"]),
                        roll_graus=float(campos["ROLL"]),
                        yaw_graus=float(campos["YAW"]),
                    )
                except (KeyError, ValueError):
                    self._mpu_value = None
                self._mpu_pending = False
                self._mpu_ready = True
                return
        if line.startswith("OK RAMPA "):
            if self._rampa_pending:
                campos = dict(
                    campo.split("=", 1)
                    for campo in line.split()[2:]
                    if "=" in campo
                )
                estado = campos.get("ESTADO")
                try:
                    angulo = float(campos["ANGULO"])
                except (KeyError, ValueError):
                    estado = None
                    angulo = None
                if estado not in ("PLANO", "SUBINDO", "DESCENDO"):
                    estado = None
                self._rampa_estado = estado
                self._rampa_angulo = angulo
                self._rampa_pending = False
                self._rampa_ready = True
                return
        if line.startswith("OK ULTRASSOM "):
            if self._ultra_pending:
                try:
                    value = int(line.split()[-1])
                    resposta_valida = value == -1 or value >= 0
                except (ValueError, IndexError):
                    value = -1
                    resposta_valida = False
                self._ultra_value = None if value < 0 else value
                self._ultra_response_received = resposta_valida
                self._ultra_pending = False
                self._ultra_ready = True
                return
            # A ferramenta manual tambem pode enviar ULTRASSOM.
            if self._manual_pending:
                self._manual_responses.append(line)
            return
        if self._manual_pending:
            self._manual_responses.append(line)
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
