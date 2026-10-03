# Case UFMG 2026/2 - Teoria da Decisão
## Otimização do custo e qualidade de pilhas de minério 

### Entrega 1 (Mono-objetivo)

### Estrutura dos arquivos
- `data.py` — parâmetros do problema (minérios, pilhas, limites de qualidade).
- `model.py` — representação da solução, funções objetivo f1/f2/f3, restrições e penalização.
- `construction.py` — heurística construtiva (geração da solução inicial).
- `neighborhoods.py` — vizinhanças N1 (troca simples), N2 (troca entre pilhas) e N3 (substituição dupla).
- `vns.py` — metaheurística GVNS (shaking + busca local + critério de aceitação).
- `experiments.py` — roda 5 execuções por objetivo, gera tabela de estatísticas, curvas de convergência e figuras da melhor solução.

### Dependências
Python 3.9+, `numpy`, `matplotlib`.

```
pip install numpy matplotlib
```

### Como executar

Reproduzir todos os resultados da Entrega 1 (tabelas, curvas de convergência e figuras):

```
python3 experiments.py
```

Os artefatos (`convergencia_f1.png`, `convergencia_f2.png`, `convergencia_f3.png`,
`melhor_solucao_f1.png`, `melhor_solucao_f2.png`, `melhor_solucao_f3.png` e
`resultados_entrega1.json`) são salvos no diretório de saída definido em
`experiments.OUT_DIR`.

Parâmetros do algoritmo (ajustáveis em `experiments.py` e `vns.py`):
- `N_RUNS = 5`, `SEEDS = [10, 20, 30, 40, 50]`
- `MAX_EVALS = 25000` (orçamento de avaliações por execução)
- `k_max = 3` (número de vizinhanças de shaking, N1–N3)
- `ls_sem_melhora = 20` (critério de parada da busca local em N1)
- `PENALTY_WEIGHT = 1e7` (peso da penalização de restrições, em `model.py`)

### Teste rápido de um único objetivo
```python
from vns import gvns
from model import resumo_solucao
sol, valor, historico = gvns("f2", seed=1, max_evals=15000)
print(resumo_solucao(sol))
```
