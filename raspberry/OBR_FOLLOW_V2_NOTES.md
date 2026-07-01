# OBR Follow v2 — Engineering Notes

## Root Cause of the Original Freeze (with evidence)

**Bug class:** Serial TX buffer deadlock (Phase B — H1 confirmed).

**Symptom:** `follow_destinos.py --camera --motores --porta auto --log` congelava após
alguns segundos. Enviar LADO fixo via serial_test.py funcionava sem problemas.

**Causa raiz, passo a passo:**

1. Pi envia `LADO vel vel\n` a cada ~33 ms (30 FPS) via `write()`. O OS bufferiza o
   envio; o `write()` retorna imediatamente.
2. Arduino recebe, executa GPIO/PWM instantaneamente (todos os comandos são
   não-bloqueantes no firmware), e responde `"OK LADO vel vel\n"` via `Serial.println()`.
3. Pi nunca lê essas respostas (`esperar_resposta=False` → sem `readline()`).
4. O buffer HW UART TX do Arduino tem **64 bytes**. Cada resposta `"OK LADO 100 80\n"`
   tem ~15 bytes — cabeça do buffer em ~4 respostas (≈60 ms de loop).
5. Quando o buffer enche, `Serial.println()` **bloqueia o loop do Arduino**. Nesse
   estado: Arduino não drena seu buffer RX, não executa o watchdog (1000 ms) e não
   processa novos comandos.
6. Pi continua escrevendo no OS serial buffer (que é maior que 64 bytes). Nas
   primeiras iterações ainda parece funcionar — as escritas vão para o OS buffer.
7. Depois de alguns segundos, o OS buffer também enche. `write()` bloqueia por
   `write_timeout=0.20s`, lança `SerialTimeoutException → RuntimeError`. A exceção
   sobe para o `except Exception` no topo de `main()`, que imprime "Erro grave:" e
   encerra.

**Por que testes manuais não repoduziam:**
- Envio manual (1–2 comandos/seg) nunca enche os 64 bytes do buffer TX.
- O loop de 30 FPS enche em ~60 ms; o OS buffer fica cheio em 2–5 s.

**Evidência no git:**
- Commit `5f213c2 (Silence continuous motor command replies)` — adicionou a flag
  `RESPONDER_COMANDOS_CONTINUOS = false` no Arduino, que silencia respostas a LADO/RODAS.
  **O freeze desapareceu com esse commit**, confirmando a causa.
- Commit `997ebb9 (Revert Arduino motor controller change)` — reverteu a mudança no
  Arduino. O freeze voltou. Como o .ino agora não pode ser modificado, a correção
  precisa ficar no lado do Pi.
- Commit `623ddd2 (Avoid blocking serial writes during motor control)` — adicionou
  `write_timeout=0.20s` e removeu `flush()` no loop. Isso **limita** o bloqueio (evita
  travar para sempre) mas **não elimina** a causa raiz, pois o buffer continua enchendo.

**Correção implementada em `obr_follow_v2.py`:**

```python
# Início de cada iteração do loop
def _drenar_serial(conexao):
    if conexao is not None and conexao.is_open:
        try:
            pendentes = conexao.in_waiting
            if pendentes > 0:
                conexao.read(pendentes)
        except Exception:
            pass
```

E envio com rate-limit (só envia quando o comando muda ou a cada 0.20 s de keepalive):

```python
def _enviar_se_necessario(conexao, cmd, ctx, agora, motores_ativo):
    mudou   = cmd != ctx["last_cmd"]
    expirou = (agora - ctx["last_send_t"]) >= OBR_V2_KEEPALIVE_INTERVAL  # 0.20 s
    if mudou or expirou:
        enviar_comando(conexao, cmd, esperar_resposta=False)
        ctx["last_cmd"] = cmd
        ctx["last_send_t"] = agora
        return True, "changed" if mudou else "keepalive"
    return False, "skip"
```

A drenagem descarta as respostas do Arduino a cada frame, mantendo o buffer TX do
Arduino sempre vazio. O rate-limit reduz o volume de envios de 30/s para ~5/s (quando
o comando é estável), dando mais folga ao buffer.

---

## O Que Mudou e Por Quê

| Aspecto | `follow_destinos.py` | `obr_follow_v2.py` |
|---|---|---|
| Freeze serial | Não corrigido (997ebb9 reverteu fix do Arduino) | Corrigido: drain + rate-limit |
| Ray-sampling | Mesmo `escolher_destino()` | Importado de `follow_destinos` |
| Controle motor | Mesmo `controlar_destino()` | Importado de `follow_destinos` |
| Estado verde | 6 modos (NORMAL…COOLDOWN_VERDE) | 5 estados spec (NO_GREEN…GREEN_COOLDOWN) |
| Validação verde | Margem de centro 8.4 px na intersecção | Semiplano do frame + estabilidade temporal |
| Detecção diagonal | Falha (8.4 px muito estreito) | Robusta (semiplano + N frames) |
| Curva 90° | tanque_90 com `memoria` dedicada | SHARP_TURN, mesmas funções importadas |
| Log por frame | Verbose ou --log simples | Linha compacta com timings e motivos |
| Ficheiro | `follow_destinos.py` (1603 linhas) | `obr_follow_v2.py` (~350 linhas) |

**Arquivos existentes não modificados** (apenas `config.py` recebeu bloco OBR_V2_*
adicionado ao final, sem remover ou alterar nada existente).

---

## Como o Estado Duplo Funciona

Dois estados ortogonais correm em paralelo a cada frame:

### Estado de Linha

| Estado | Entrada | Saída | Comando |
|---|---|---|---|
| `LINE_FOLLOW` | Startup; linha reencontrada | Sem linha → `LOST_LINE` ou `GAP_CROSS`; curva 90° → `SHARP_TURN` | `LADO vel_e vel_d` via `controlar_destino()` |
| `SHARP_TURN` | Condição tanque_90 | Frente confiável após tempo mín, ou timeout | `GIRAR_ESQ/DIR` + `TANQUE_90_VEL` |
| `GAP_CROSS` | Perda breve (<0.30 s) em `FRENTE` | Linha encontrada ou timeout 0.50 s | `LADO VEL_RECUPERAR VEL_RECUPERAR` (reto) |
| `LOST_LINE` | Perda longa (>0.30 s) | Linha encontrada ou timeout 3.00 s → `SAFE_STOP` | Varredura `GIRAR_ESQ/DIR` alternada |
| `SAFE_STOP` | Timeout 3 s em `LOST_LINE` | Terminal | `PARAR` |

### Estado Verde (overlay)

| Estado | Entrada | Saída | Prioridade de comando |
|---|---|---|---|
| `NO_GREEN` | Default; saída de `GREEN_COOLDOWN` | 1 frame com verde → `GREEN_CANDIDATE` | Defer ao estado de linha |
| `GREEN_CANDIDATE` | 1 frame verde | N=4 frames concordantes → `GREEN_CONFIRMED`; mudança ou timeout → `NO_GREEN` | Defer ao estado de linha |
| `GREEN_CONFIRMED` | N frames acumulados | Após 1.50 s → `GREEN_ACTION` | `controlar_destino(confirmar=True)` (cauteloso) |
| `GREEN_ACTION` | De `GREEN_CONFIRMED` | Linha após tempo mín, ou timeout | **Sobrepõe linha**: `GIRAR_ESQ/DIR` |
| `GREEN_COOLDOWN` | De `GREEN_ACTION` | 3 frames sem verde + 2.00 s → `NO_GREEN` | Defer ao estado de linha |

**`GREEN_ACTION` tem prioridade sobre todos os estados de linha exceto `SAFE_STOP`.**
Durante `GREEN_ACTION`, o timer de `LOST_LINE` é reiniciado a cada frame para evitar
que o robo caia em `SAFE_STOP` durante manobras longas (RETORNO = 4.5 s).

### Validação Verde — Por Que Funciona na Abordagem Diagonal

O problema original: `verdes.py` exige que o centro do verde esteja a ≤ 8.4 px do
centro da linha na intersecção. Quando o robô sai de uma curva ligeiramente inclinado,
o centro do marcador verde desloca 10–20 px, falha o check e a detecção é rejeitada.

Solução: em vez de verificar alinhamento com a linha, verificamos apenas **em que
semiplano do frame** o verde aparece. Um marcador claramente à esquerda (cx < 40% da
largura) ou à direita (cx > 60%) é aceito. Marcadores exatamente no centro são
rejeitados como ambíguos. A **estabilidade temporal** (N=4 frames consecutivos
concordando) filtra falsos positivos sem precisar de alinhamento preciso.

Isso funciona porque, mesmo abordando diagonalmente, o robô vê o verde claramente
no semiplano correto. Os poucos frames de abordagem diagonal que podem ter o verde
na zona morta central simplesmente não contribuem para o contador — o próximo frame,
com o robô já mais alinhado, contribui normalmente.

---

## Sequência de Teste Segura no Robô Real

### 1 — Sem motores (verificar visão e log)

```
python3 -u raspberry/obr_follow_v2.py --camera --log
```

Confirmar:
- Linha de log por frame aparece, `loop_ms` ≤ 40 ms sistematicamente
- `lstate=LINE_FOLLOW` quando há linha visível
- `gstate` avança para `GREEN_CANDIDATE` ao mostrar cartão verde
- `gstate=GREEN_CONFIRMED` após 4 frames concordantes
- `Ctrl+C` termina limpo: imprime `[INFO] Encerrado.` sem traceback

### 2 — Motores ligados, robô elevado (rodas girando no ar)

```
python3 -u raspberry/obr_follow_v2.py --camera --motores --porta auto --log
```

Elevar o robô em suporte/blocos antes de ligar motores. Confirmar:
- Loop roda por 60 s mínimos sem travar nem imprimir `[ERRO]`
- Log mostra `sent=1 reason=keepalive` a cada ~0.20 s quando comando é estável
- Log mostra `sent=0 reason=skip` entre keepalives
- Rodas giram de acordo com o que o log diz
- `Ctrl+C`: robô para em ≤ 0.20 s (próximo keepalive + watchdog do Arduino)

### 3 — Pista real (só após passo 2 limpo por 60 s)

Iniciar com velocidades conservadoras: reduzir `DEST_BASE_FRENTE` e `DEST_BASE_CURVA`
em config.py para ≈ 60% dos valores padrão na primeira corrida.

### 4 — Testar marcadores verdes (diagonal)

Segurar cartão verde a ~30 cm da câmera em ângulo de ~20° de inclinação lateral.
Confirmar que `gstate` avança normalmente para `GREEN_CANDIDATE` e `GREEN_CONFIRMED`
mesmo com o marcador fora de eixo.

---

## Parâmetros para Ajustar Primeiro

Em ordem de impacto:

| Parâmetro | Localização | Direção para ajuste |
|---|---|---|
| `OBR_V2_GREEN_STABILITY_N` | config.py | Diminuir para 3 se detecção for lenta; aumentar para 5 se houver falsos positivos |
| `OBR_V2_GREEN_MARGIN_MULT` | config.py | Diminuir (0.05) se verde na borda for rejeitado; aumentar (0.15) se falsos positivos |
| `OBR_V2_MAX_GREEN_ACT_TIME` | config.py | Aumentar se curva não completa; diminuir se robo passa além |
| `DEST_BASE_FRENTE` e `DEST_BASE_CURVA` | config.py | Ajustar velocidades base na pista |
| `OBR_V2_KEEPALIVE_INTERVAL` | config.py | Manter < 1.0 s (watchdog Arduino); 0.20 s é seguro |
| `OBR_V2_GAP_MAX_LOSS_TIME` | config.py | Aumentar se lacunas normais caem em LOST_LINE antes de GAP_CROSS |

---

## Fontes e Referências

A lógica de ray-sampling, controle proporcional por destino e detecção tanque_90
foram integralmente reaproveitadas de `follow_destinos.py` (sem modificação). A
detecção de cor verde foi reaproveitada de `verdes.py` (pipeline HSV + BGR), substituindo
apenas a lógica de validação posicional.

A abordagem de drain de buffer serial para corrigir deadlock de UART bidirecional é
técnica padrão de comunicação serial em tempo real — ver, por exemplo:
- pyserial docs: `Serial.in_waiting` + `Serial.read(n)` para drenagem não-bloqueante
- Arduino Forum: "Arduino Serial TX buffer full blocks loop" (recorrente em projetos
  de controle motor + Python)

A estabilidade temporal (N frames consecutivos) como substituta de checks geométricos
rígidos é abordagem comum em pipelines de visão para robótica: reduz taxa de
falso-negativo em condições de abordagem variável sem aumentar falso-positivo, desde
que N seja calibrado para o intervalo de frame da aplicação.
