# RUNBOOK — Shadow2026 line follower, do boot zero à linha

Guia autocontido: alguém com o mesmo hardware e nenhum contexto deve conseguir
seguir este documento até ter um robô seguindo linha. Siga as seções **em ordem**.

---

## 1. Checklist de materiais

- [ ] Raspberry Pi 5 (8 GB) com Raspberry Pi OS Bookworm (Python 3.11)
- [ ] Arduino Uno com cabo USB conectado ao Pi
- [ ] 2× driver TB6612FNG (um por lado)
- [ ] 4× motor DC 12 V com redução, rodas ~5 cm (FE, TE, FD, TD)
- [ ] Câmera CSI (Picamera2), montada a ~8 cm do chão, ~35° para baixo
- [ ] LiPo 7.4 V 2200 mAh + regulador step-down 5 V para o Pi
- [ ] TB6612 VM ligado direto na LiPo; Uno alimentado pelo USB do Pi
- [ ] **Todos os GNDs comuns** (LiPo, regulador, Pi, Uno, 2× TB6612)
- [ ] Fiação conforme a seção 2 (pinos travados do projeto)
- [ ] Este repositório clonado em `~/Overengineering-squared-RoboCup` e o
      repositório Shadow2026 (firmware) ao lado, em `~/Shadow2026`

## 2. Diagrama de fiação (Uno → TB6612 → motores)

```
                         ARDUINO UNO
        ┌────────────────────────────────────────────┐
        │  A1  A0  D9      A2  A3  D10               │
        │   │   │   │       │   │   │                │
        │  IN1 IN2 PWM     IN1 IN2 PWM               │
        └───┼───┼───┼───────┼───┼───┼────────────────┘
            │   │   │       │   │   │
       ┌────▼───▼───▼───────▼───▼───▼────┐          ┌── LiPo 7.4 V ──┐
       │  B: FE (frente esq)  A: TE (trás│          │  VM ◄──────────┤
       │  TB6612 ESQUERDO          esq)  │◄─ VM ────┤  (também no    │
       │  GND ◄── comum                  │          │   TB6612 dir.) │
       └─────────────────────────────────┘          └────────────────┘
            D3  D2  D5      D4  D7  D6
             │   │   │       │   │   │
            IN1 IN2 PWM     IN1 IN2 PWM
       ┌────▼───▼───▼───────▼───▼───▼────┐
       │  B: FD (frente dir)  A: TD (trás│
       │  TB6612 DIREITO           dir)  │
       │  GND ◄── comum                  │
       └─────────────────────────────────┘

  GND comum: LiPo ─ regulador 5 V ─ Pi ─ Uno ─ TB6612 esq ─ TB6612 dir
  Pi 5 ◄── 5 V do regulador        Uno ◄── USB do Pi (dados + alimentação)
  STBY de cada TB6612 ── VCC lógico (habilitado)
  NUNCA usar D0/D1 do Uno (serial USB)
```

## 3. Setup único (no Pi)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-gpiozero
python3 -m pip install -r shadow/requirements.txt --break-system-packages
```

Rode os comandos acima manualmente (o stack nunca chama `sudo` sozinho).

## 4. Firmware do Arduino (verificar / gravar)

O Uno usa o firmware **SPEC 01 já existente** no repositório Shadow2026 —
o `shadow/` não traz firmware próprio.

1. Verifique se já está gravado: abra o Serial Monitor (Arduino IDE ou
   `python3 -m shadow.tools.serial_smoke`) a **115200** e pressione reset —
   deve aparecer `Arduino pronto - SPEC 01` uma única vez.
2. Se não aparecer: abra
   `~/Shadow2026/arduino/motor_controller/motor_controller.ino`
   no Arduino IDE, placa **Arduino Uno**, porta `/dev/ttyACM0` (ou a que
   aparecer), **Upload**. Confirme o banner no Serial Monitor.
3. Feche o Serial Monitor antes de rodar qualquer script (a porta é exclusiva).

## 5. Checagem de polaridade dos motores — **não pule**

Com as **rodas suspensas** (robô sobre um apoio):

```bash
cd ~/Overengineering-squared-RoboCup
python3 -m shadow.tools.serial_smoke
```

O ciclo é: para → **frente** 2 s → para → **ré** 2 s → para (PWM 60).
Para **cada roda** que girar ao contrário do anunciado:

1. Abra `~/Shadow2026/arduino/motor_controller/config.h`
2. Troque o multiplicador da roda: `#define DIRECAO_FE 1` → `#define DIRECAO_FE -1`
   (idem `DIRECAO_TE`, `DIRECAO_FD`, `DIRECAO_TD`)
3. Regrave o firmware (seção 4) e repita o teste até as 4 rodas obedecerem.

Depois valide o watchdog:

```bash
python3 -m shadow.tools.serial_smoke --watchdog
```

Os motores devem parar **sozinhos ~1 s** após o script silenciar. Se não
pararem, NÃO prossiga — confira o firmware.

Opcional: dirija com o teclado — `python3 -m shadow.tools.steer_test`
(w/a/s/d, espaço para parar, x para sair).

## 6. Calibração de cores

```bash
python3 -m shadow.tools.color_slider
```

Com o robô sobre a pista, na iluminação real (precisa de monitor ou X11):

- **Teclas 1/2** — tetos BGR do preto (faixa distante 0-40 % / próxima
  40-100 % da imagem, como no Hotspot 1 do OE²). "Bom" = linha branca sólida
  na máscara, fundo totalmente preto.
- **Tecla 3** — teto de rampa (bem escuro; usado só quando "escuro à frente").
- **Tecla 4** — HSV do verde: marcador branco sólido, resto preto.
- **Teclas 5/6** — HSV do vermelho (duas bandas de hue: 0-10 e 170-180).
- **`s` salva o grupo atual** em `shadow/config.ini`; `q` sai.

Detalhes e critérios em [docs/CALIBRATION_GUIDE.md](docs/CALIBRATION_GUIDE.md).
Cheque a câmera isolada antes, se quiser: `python3 -m shadow.tools.camera_smoke`
(salva um JPEG 448×252 e mede o FPS — alvo 40).

## 7. Primeiro teste de direção (suspenso)

Rodas suspensas, linha impressa na mão:

```bash
python3 shadow/main.py --debug
```

- Janela mostra a linha detectada (contorno azul), o POI (círculo vermelho)
  e `ang=` no rodapé, estável a ~35-40 fps.
- Linha centralizada → `ang≈0`, as 4 rodas giram para frente por igual.
- Linha deslocada p/ **direita** da imagem → `ang` positivo → roda **direita**
  desacelera (o robô "viraria" para a direita). Espelhado para a esquerda.
- Linha muito ao lado → |ang| > 110 → as rodas contra-giram (pivot).

Se girar ao contrário do esperado, revise a seção 5 antes de ir ao chão.
(Primeira execução compila o Numba — pode demorar ~10-20 s extra, só na 1ª vez.)

## 8. Primeiro teste no chão

Coloque o robô numa linha reta e rode `python3 shadow/main.py --debug`.

- Deve seguir a reta sem oscilar mais que ±3 cm.
- **Oscilando (zigue-zague)?** Suba `max_turn_angle` em `shadow/config.py`
  (padrão 110 do dossiê; tente 120-130) — isso suaviza a resposta.
- **Curvas moles / perde a linha em 90°?** Desça `max_turn_angle`.
- **Puxa para um lado em reta?** Ajuste `left_correction`/`right_correction`.

## 9. Operação autônoma (o comando final)

```bash
cd ~/Overengineering-squared-RoboCup
python3 shadow/main.py
```

O que acontece: ~2 s inicializando câmera e serial (a 1ª execução pós-boot
compila o cache do Numba e demora mais), aparece
`Shadow2026 ready — awaiting line`, e o robô passa a seguir qualquer linha sob
a câmera. Status em português no terminal (`Seguindo Linha`,
`Orientando no gap`, `Parada por vermelho: N s restantes`…).
**Ctrl-C para tudo de forma limpa** (PARAR é enviado ao Uno na saída).
Headless: funciona sem monitor/display.

## 10. Execução com debug

```bash
python3 shadow/main.py --debug
```

Igual à operação normal + janela única com o frame anotado (contorno, POI,
geometria do gap, retângulo do vermelho, fps, status). `q` na janela ou
Ctrl-C encerra. Há também `--vision-only --debug` (só visão, motores parados).

## 11. Autostart no boot (opcional)

Crie `/etc/systemd/system/shadow-line.service`:

```ini
[Unit]
Description=Shadow2026 line follower
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Overengineering-squared-RoboCup
ExecStart=/usr/bin/python3 shadow/main.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reloadD
sudo systemctl enable --now shadow-line.service   # ativa
sudo systemctl disable shadow-line.service        # desativa
journalctl -u shadow-line -f                      # logs
```

(Ajuste `User=`/caminhos se seu usuário não for `pi`.)

## 12. Troubleshooting

| Sintoma | Causa provável | Onde mexer |
|---|---|---|
| `Arduino nao encontrado` no boot | Cabo USB / firmware sem banner / Serial Monitor aberto | Seção 4; feche o IDE; confira `ls /dev/ttyACM*` |
| Motores não giram, serial OK | STBY do TB6612 solto; VM sem bateria; GND não comum | Seção 2 |
| Uma roda gira ao contrário | Polaridade | `DIRECAO_*` no `config.h` do Shadow2026 (seção 5) |
| Motores param sozinhos andando | Watchdog disparando → loop Python travado/lento | Veja o terminal; reporte — não deveria ocorrer com `sleep_steering` |
| Câmera não encontrada | Cabo CSI; picamera2 ausente | `python3 -m shadow.tools.camera_smoke`; seção 3 |
| Linha não detectada (máscara vazia) | Tetos de preto baixos | `color_slider` grupos 1/2 |
| Robô lento sem motivo, círculo preto no canto do --debug | `ramp_ahead` disparando com chão fora da pista (fish-eye) | Suba `RAMP_SWAP_TRIGGER` (90 → 110-130) em `config.py` |
| "Perde" a linha na descida de rampa | Teto de rampa errado | `color_slider` grupo 3 |
| Robô pivota para o lado errado no verde | Marcador validado com a linha errada — máscara verde suja ou preta fraca | `color_slider` grupos 1/2/4; confira no `--debug` |
| Verde ignorado | Área < 2500 px² ou vizinhança preta não bate | `GREEN_MIN_AREA`; recalibre preto+verde |
| 180° passa/falta do alvo | `T_180` fora | Cronometre e ajuste `T_180` em `config.py` |
| Vermelho nunca para | Contorno < 15000 px² | Desça `RED_MIN_CONTOUR`; recalibre grupos 5/6 |
| Para em vermelho "fantasma" | HSV de vermelho largo (pele/madeira) | Aperte S/V mínimos nos grupos 5/6 |
| Gap aborta sempre ("Validação falhou") | `black_average > 40` — máscara preta suja | Grupos 1/2; `GAP_BLACK_AVG_MAX` |
| Oscila em linha reta | Ganho alto | Suba `max_turn_angle` (seção 8) |
| 1ª execução muito lenta | Compilação Numba (uma vez) | Esperar; cache persiste |

## 13. O que NÃO está implementado (fora de escopo)

Herdado da lista do dossiê (Seção 8) — pontos de extensão futuros:

- Faixa prateada / entrada da zona de resgate (silver AI)
- Zona de evacuação, vítimas, YOLO/TPU, segunda câmera
- IMU (rampas por giroscópio, giros com feedback de yaw)
- Obstáculos e sensores IR de distância; gangorra (seesaw)
- Garra/servos, kit de resgate, LEDs, GUI touchscreen

O comportamento nas rampas é apenas o ramo de câmera (`ramp_ahead`); giros de
90°/180° são por visão/tempo, sem giroscópio.
