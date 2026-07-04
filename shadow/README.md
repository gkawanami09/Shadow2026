# shadow/ — Segue-linha Shadow2026 (port do Overengineering²)

Port completo do stack de segue-linha do **Overengineering²** (campeão mundial
RoboCup Junior Rescue Line, Eindhoven 2024) para o hardware do **Shadow2026**:

- Raspberry Pi 5 (8 GB), Raspberry Pi OS Bookworm, Python 3.11
- 1× Arduino Uno via USB serial (115200) com firmware **SPEC 01** já existente
- 4× motores DC 12 V via 2× TB6612FNG (2 motores por lado)
- Câmera CSI (Picamera2), fish-eye 160°, 8 cm do chão, 35° de inclinação
- LiPo 7.4 V 2200 mAh

**Sem** IMU, TPU, sensores IR, segunda câmera, garra ou GUI.

## O que ele faz

1. **Segue linha preta** — lei P-only do OE² (POI → `steer(angle, speed)`)
2. **Cruza gaps** — valida, alinha em até 7 ciclos e cruza às cegas
3. **Marcadores verdes** — 90° esquerda/direita e 180° no verde duplo
4. **Linha vermelha** — para 9 s

## Comece por aqui

➡️ **[RUNBOOK.md](RUNBOOK.md)** — passo a passo do boot zero até o robô
seguindo linha (instalação, firmware, polaridade, calibração, testes).

## Rodando

```bash
cd ~/Overengineering-squared-RoboCup
python3 shadow/main.py            # operação normal (headless)
python3 shadow/main.py --debug    # com janela da câmera anotada
```

## Documentação

| Arquivo | Conteúdo |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **O guia operacional completo** |
| [MAPEAMENTO.md](MAPEAMENTO.md) | Mapa função-a-função OE² → shadow/ (PORT/ADAPT/REBUILD/SKIP) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 2 processos, variáveis compartilhadas, máquina de estados |
| [docs/DIFFERENCES_FROM_OE2.md](docs/DIFFERENCES_FROM_OE2.md) | Cada desvio do OE², com motivo |
| [docs/CALIBRATION_GUIDE.md](docs/CALIBRATION_GUIDE.md) | Cores, polaridade, constantes a retunar |
| [serial_link/PROTOCOL.md](serial_link/PROTOCOL.md) | Protocolo SPEC 01 Pi ⇄ Uno |

A fonte algorítmica é o dossiê `../OE2_READING_DOSSIER.md`; cada arquivo
portado cita o hotspot e as linhas de origem no cabeçalho.
