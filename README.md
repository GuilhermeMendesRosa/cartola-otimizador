# Cartola Otimizador

Monta a melhor escalação do Cartola FC usando otimização matemática com base no seu orçamento de cartoletas.

## Instalação

```bash
cd ~/code/cartola-otimizador
pip3 install -r requirements.txt
```

## Fluxo de uso a cada rodada

Sempre que o mercado abrir (quinta-feira), siga esses **3 passos**:

### 1. Coletar dados atualizados

```bash
python3 -m src.main collect
```

Isso puxa os dados da rodada atual da API do Cartola (preços, médias, status dos atletas, partidas)
e salva localmente. Na primeira vez, adicione um número de rodadas históricas:

```bash
python3 -m src.main collect --rodadas 10
```

### 2. (Opcional) Ver status do mercado

```bash
python3 -m src.main status
```

Mostra rodada atual, se o mercado está aberto, times escalados, etc.

### 3. Montar a melhor escalação

```bash
python3 -m src.main scale --cartoletas 120 --formacao 4-3-3
```

Troque `120` pelo seu saldo de cartoletas e escolha a formação desejada.

---

## Comandos

### `scale` — Encontrar melhor escalação

```bash
python3 -m src.main scale [OPÇÕES]
```

| Opção | Padrão | Descrição |
|---|---|---|
| `-c, --cartoletas` | 100 | Seu saldo de cartoletas |
| `-f, --formacao` | 4-3-3 | Formação: `4-3-3`, `3-4-3`, `3-5-2`, `4-4-2`, `5-3-2` |
| `-i, --incluir` | — | Slug de atleta obrigatório (pode repetir) |
| `-e, --excluir` | — | Slug de atleta banido (pode repetir) |
| `-m, --max-clube` | 4 | Máximo de atletas do mesmo clube |

**Exemplos:**

```bash
# 115.94 cartoletas, 4-3-3
python3 -m src.main scale -c 115.94 -f 4-3-3

# 105 cartoletas, 3-5-2, forçando Hulk e banindo Gabigol
python3 -m src.main scale -c 105 -f 3-5-2 -i hulk -e gabigol

# Time com no máximo 3 jogadores do mesmo clube
python3 -m src.main scale -c 100 -f 4-4-2 -m 3
```

### `collect` — Atualizar dados

```bash
python3 -m src.main collect [OPÇÕES]
```

| Opção | Descrição |
|---|---|
| `-r, --rodadas` | Número de rodadas históricas para baixar (só precisa na 1ª vez) |

### `status` — Status do mercado

```bash
python3 -m src.main status
```

---

## Como funciona

1. **Coleta** — Puxa dados da API oficial do Cartola (atletas, preços, médias, partidas)
2. **Predição** — Calcula pontuação esperada de cada atleta considerando:
   - Média ponderada das últimas rodadas (recentes pesam mais)
   - Mediana (robusto a outliers)
   - Momento/tendência (em alta ou baixa nas últimas rodadas)
   - Jogo em casa (+1.2 pts) ou fora (-0.5 pts)
   - Força do adversário (z-score por clube e posição)
3. **Otimização** — Resolve um problema de programação linear inteira (ILP) que maximiza
   a pontuação esperada respeitando:
   - Orçamento de cartoletas
   - Quantidade de jogadores por posição (formação)
   - Limite de atletas por clube
   - Exclusão de lesionados/suspensos

## Estrutura do projeto

```
cartola-otimizador/
├── data/
│   ├── raw/              # JSON bruto da API por rodada
│   └── processed/        # CSV consolidado com histórico
├── src/
│   ├── main.py           # CLI
│   ├── api.py            # Consumidor da API do Cartola
│   ├── data_collector.py # Coleta e processa dados
│   ├── predictor.py      # Modelo de predição de pontuação
│   ├── optimizer.py      # Solver ILP (PuLP/CBC)
│   └── formatter.py      # Output formatado no terminal
└── requirements.txt
```
