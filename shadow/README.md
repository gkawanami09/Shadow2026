# shadow/ — Segue-linha Shadow2026 (port do Overengineering²)

Port completo do stack de segue-linha do **Overengineering²** (campeão mundial
RoboCup Junior Rescue Line, Eindhoven 2024) para o hardware do **Shadow2026**:

- Raspberry Pi 5 (8 GB), Raspberry Pi OS Bookworm, Python 3.11
- 1× Arduino Uno via USB serial (115200) com firmware **SPEC 01** já existente
- 4× motores DC 12 V via 2× TB6612FNG (2 motores por lado)
- Câmera CSI de segue-linha, fish-eye 160°, 8 cm do chão, 35° de inclinação
- Câmera CSI frontal exclusiva do resgate
- LiPo 7.4 V 2200 mAh

Mapeamento das câmeras no Pi 5: resgate no índice `0`; segue-linha no flat 2,
índice `1`. Os dois programas abrem seus índices explicitamente.

O executável de segue-linha continua **sem** IMU, TPU e sensores IR. O resgate
fica isolado em `rescue_main.py` e usa somente a câmera frontal. Durante a
aproximação, o detector mantém um círculo temporal preso à mesma esfera e
confirma a distância quando ele cobre o `PONTO GARRA` perto da base da imagem.
Se o círculo for cortado logo depois do contato inferior, a borda larga em
meia-lua pode fornecer somente a segunda confirmação. O primeiro contato
sempre precisa vir do círculo rastreado. Depois o robô baixa o Futaba, avança
por 2 s e fecha as garras durante esse avanço, sem executar ré e sem alterar
`main.py`. A câmera de linha e o HC-SR04 não participam de nenhuma decisão do
resgate.

## O que ele faz

1. **Segue linha preta** — lei P-only do OE² (POI → `steer(angle, speed)`)
2. **Cruza gaps** — valida, alinha em até 7 ciclos e cruza às cegas
3. **Marcadores verdes** — 90° esquerda/direita e 180° no verde duplo
4. **Linha vermelha** — para 9 s
5. **Resgate separado** — encontra e rastreia a esfera até ela tocar o ponto
   inferior da garra, baixa o Futaba, avança por 2 s e fecha as duas garras
   durante o avanço, sem dar ré

## Comece por aqui

➡️ **[RUNBOOK.md](RUNBOOK.md)** — passo a passo do boot zero até o robô
seguindo linha (instalação, firmware, polaridade, calibração, testes).

## Rodando

```bash
cd ~/Overengineering-squared-RoboCup
python3 shadow/main.py            # operação normal (headless)
python3 shadow/main.py --debug    # com janela da câmera anotada
python3 shadow/rescue_main.py --camera-index 0 --drive --debug
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
