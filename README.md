# Robô Seguidor de Linha — OBR Prática

Projeto de ensino médio para desenvolver um robô seguidor de linha para a OBR Prática.

A Raspberry Pi 5 executará os programas em Python, fará a captura da câmera e, nas fases futuras, tomará decisões a partir da imagem. O Arduino Uno controlará os motores. Os dois dispositivos se comunicarão por USB Serial.

## Estado atual

As Fases 1, 2 e 3 estão prontas para testes físicos: a Raspberry envia comandos pela USB Serial ao Arduino, captura imagens da câmera CSI e detecta a linha preta em uma imagem. Ainda não há movimento automático do robô.

Por segurança, os motores iniciam parados, a velocidade é limitada a 120 e o Arduino para todos os motores após 1 segundo sem receber um comando completo.

## Próximas fases

1. Segue-linha inicial.
2. Detecção de verde e vermelho.
3. Obstáculos e ajustes para a OBR.

## Uso inicial

Na Raspberry Pi, instale as dependências com `pip install -r requirements.txt` e execute `python raspberry/main.py`.

Envie `arduino/motor_controller/motor_controller.ino` ao Arduino. Com o robô suspenso, teste a comunicação com:

```bash
python3 raspberry/serial_test.py --porta auto --comando PING
```

Para testar cada motor, use `python3 raspberry/serial_test.py --porta auto --teste-motores`. Mantenha as rodas fora do chão durante esse teste.

Para salvar uma imagem da câmera CSI, use `python3 raspberry/camera_test.py`. As imagens de teste são salvas em `captures/`.

Para detectar a linha em uma imagem salva, use `python3 raspberry/line_test.py --imagem captures/NOME_DA_IMAGEM.jpg --salvar-mascara`.

## Follow Destinos 1.0

O controlador experimental `follow_destinos.py` escolhe um destino visual por raios, com prioridade para frente, depois curva e por fim retorno. O `follow_clean.py` continua disponivel como controlador de backup.

```bash
python3 raspberry/follow_destinos.py --camera --salvar-debug
python3 raspberry/follow_destinos.py --camera --motores --porta auto --salvar-debug
```

Teste primeiro sem motores e depois com o robo suspenso. A parada normal e `CTRL+C`.
