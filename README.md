# Robô Seguidor de Linha — OBR Prática

Projeto de ensino médio para desenvolver um robô seguidor de linha para a OBR Prática.

A Raspberry Pi 5 executará os programas em Python, fará a captura da câmera e, nas fases futuras, tomará decisões a partir da imagem. O Arduino Uno controlará os motores. Os dois dispositivos se comunicarão por USB Serial.

## Estado atual

As Fases 1 e 2 estão prontas para testes físicos: a Raspberry envia comandos pela USB Serial ao Arduino e também pode capturar imagens da câmera CSI. Ainda não há visão computacional ou segue-linha.

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
