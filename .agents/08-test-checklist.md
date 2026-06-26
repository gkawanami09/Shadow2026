# Checklist de testes

Antes de considerar uma alteração pronta, testar:

## Segue-linha
- [ ] Linha reta.
- [ ] Curva suave para esquerda.
- [ ] Curva suave para direita.
- [ ] Curva fechada para esquerda.
- [ ] Curva fechada para direita.
- [ ] Linha parcialmente fora do centro.
- [ ] Variação de iluminação.
- [ ] Sombra na pista.

## Verde
- [ ] Verde à esquerda.
- [ ] Verde à direita.
- [ ] Verde dos dois lados.
- [ ] Verde pequeno falso.
- [ ] Objeto verde fora da pista.
- [ ] Reflexo parecido com verde.
- [ ] Verde perto da linha.
- [ ] Verde longe da linha.

## Gap
- [ ] Linha some por pouco tempo.
- [ ] Gap real com continuação à frente.
- [ ] Linha perdida por erro do robô.
- [ ] Beco/fim de caminho.
- [ ] Robô não gira imediatamente ao perder a linha.
- [ ] Robô não acelera durante busca.

## Vermelho
- [ ] Vermelho real na pista.
- [ ] Pequeno ruído vermelho.
- [ ] Objeto vermelho fora da pista.
- [ ] Reflexo vermelho.
- [ ] Parada pelo tempo correto.

## Segurança
- [ ] Motores param quando necessário.
- [ ] Não há giro infinito.
- [ ] Não há loop travado.
- [ ] O robô consegue voltar para LINE_FOLLOW depois de uma recuperação.