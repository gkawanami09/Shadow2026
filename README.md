# Shadow2026

Código do robô Shadow para a OBR. A pasta principal é `shadow/`.

O comando autônomo da prova é o controlador central:

```bash
python3 shadow/mission.py
```

Ele permanece ativo do boot ao vermelho final e coordena, sem abrir outro
programa, as duas rotinas:

- `shadow/main.py`: segue a linha, lê verde e vermelho e trata gap;
- `shadow/resgate.py`: encontra a vítima, aproxima e executa a coleta.

Enquanto segue a linha, a câmera frontal também mantém o YOLO de vítimas
ativo. Uma vítima plausível confirmada em vários frames e com confiança de ao
menos 0,80 faz o controle parar os motores e iniciar o resgate. Após uma
reinicialização do robô — inclusive se ela ocorreu no resgate — a missão sobe
novamente no modo segue-linha com esse vigia já ativo.

`main.py` e `resgate.py` são entradas de diagnóstico isolado. Não os execute
ao mesmo tempo nem em paralelo com `mission.py`, pois usam as câmeras, os
motores e a mesma conexão serial.

## Instalação

Na Raspberry Pi:

```bash
python3 -m pip install -r shadow/requirements.txt --break-system-packages
```

`picamera2` e `libcamera` devem ser instalados pelos pacotes do Raspberry Pi OS,
como explicado em `shadow/RUNBOOK.md`.

## Diagnóstico do segue-linha

```bash
python3 shadow/main.py
python3 shadow/main.py --debug
python3 shadow/main.py --vision-only --debug
```

## Diagnóstico do resgate

Primeiro teste sem liberar os motores:

```bash
python3 shadow/resgate.py --camera-index 0 --debug
```

Depois da conferência visual:

```bash
python3 shadow/resgate.py --camera-index 0 --drive --debug
```

## Ferramentas

```bash
python3 -m shadow.tools.calibrar_cores
python3 -m shadow.tools.visualizar_cameras
python3 -m shadow.tools.teste_camera
python3 -m shadow.tools.teste_entrada_onnx
python3 -m shadow.tools.teste_serial
python3 -m shadow.tools.teste_direcao
python3 -m shadow.tools.controle_serial
```

## Arduino

O firmware usado pelo robô continua sendo:

```text
arduino/motor_controller/motor_controller.ino
```

O pinout e o protocolo não foram alterados pela limpeza do projeto.

## Testes

```bash
python3 -m unittest discover -s shadow/tests -p "test_*.py" -v
```

Mais detalhes estão em `shadow/README.md` e `shadow/RUNBOOK.md`.
