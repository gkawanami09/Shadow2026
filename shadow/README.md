# Código principal do Shadow2026

Esta pasta contém todo o código executado pela Raspberry Pi.

## Hardware

- Raspberry Pi 5;
- Arduino Uno com o firmware SPEC 01;
- quatro motores controlados por dois TB6612FNG;
- câmera de resgate no índice `0`;
- câmera de segue-linha no índice `1`;
- garras e elevador controlados pelo Arduino.

## Programas

Para a missão autônoma completa, use:

```bash
python3 shadow/mission.py
```

`mission.py` é a única autoridade de estado e executa percurso, entrada,
resgate, saída e retorno à linha. Para diagnosticar apenas o segue-linha:

```bash
python3 shadow/main.py
python3 shadow/main.py --debug
```

Para diagnosticar apenas o resgate:

```bash
python3 shadow/resgate.py --camera-index 0 --debug
python3 shadow/resgate.py --camera-index 0 --drive --debug
```

Na busca, uma soleira confiável veta alvos fora da arena; filtros independentes
rejeitam madeira, sombras e triângulos. Depois da coleta, prata é levada ao
triângulo verde e preta ao vermelho; o movimento de garra só é liberado depois
da chegada confirmada.

Os diagnósticos não podem rodar juntos nem em paralelo com `mission.py` porque
compartilham os motores e a conexão serial.

## Organização

| Caminho | Conteúdo |
|---|---|
| `controle/` | decisões e movimentos |
| `visao/` | câmeras e detectores |
| `comunicacao_serial/` | comunicação com o Arduino |
| `shared/` | dados compartilhados entre processos |
| `tools/` | calibração e testes manuais |
| `tests/` | testes automáticos |
| `docs/` | explicações do projeto |

## Documentação

- [RUNBOOK.md](RUNBOOK.md): instalação e testes no robô;
- [docs/ARQUITETURA.md](docs/ARQUITETURA.md): organização dos processos;
- [docs/GUIA_CALIBRACAO.md](docs/GUIA_CALIBRACAO.md): calibração da pista;
- [docs/PLANO_BOLA_RESGATE.md](docs/PLANO_BOLA_RESGATE.md): visão e coleta;
- [comunicacao_serial/PROTOCOLO.md](comunicacao_serial/PROTOCOLO.md): comandos
  trocados com o Arduino.
