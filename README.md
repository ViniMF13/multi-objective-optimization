# Case UFMG 2026/2 - Teoria da Decisão
## Otimização de Mistura e Alocação de Pilhas de Minério de Ferro para Sinterização

Este repositório contém a modelagem matemática, metaheurísticas e resultados experimentais da **Etapa 1 (Otimização Mono-Objetivo e Fundamentos de Teoria da Decisão Multiobjetivo)**, com a preparação completa para a **Etapa 2 (Otimização Multiobjetivo e Análise de Decisão)**.

---

## 1. Descrição do Problema e Modelagem Matemática

O problema consiste em determinar a composição ótima de **10 pilhas de homogeneização de minério de ferro** (totalizando 290.000 toneladas / 145 caminhões de 2 kt cada), destinadas ao abastecimento de duas plantas de sinterização industriais (**Sinter 1** e **Sinter 2**):

- **Sinter 1 (Pilhas P1 a P5):** 60 caminhões (120 kt). Limites químicos: $\text{SiO}_2 \in [5,0; 6,2]\%$ (alvo operacional: $5,55\%$), $\text{Al}_2\text{O}_3 \in [1,0; 2,3]\%$ (alvo operacional: $1,25\%$).
- **Sinter 2 (Pilhas P6 a P10):** 85 caminhões (170 kt). Limites químicos: $\text{SiO}_2 \in [5,0; 6,5]\%$ (alvo operacional: $5,50\%$), $\text{Al}_2\text{O}_3 \in [1,0; 2,2]\%$ (alvo operacional: $1,50\%$).
- **Disponibilidade e Pátio:** 20 tipos de minérios (M1 a M20), cada um com limites operacionais de disponibilidade mínima ($\text{disp\_min}$) e máxima ($\text{disp\_max}$), regras estritas de elegibilidade tecnológica por equipamento (S1 e/ou S2) e custo de aquisição unitário (R$/t).

### Funções Objetivo
1. **$f_1$ (Custo Financeiro Total de Aquisição - R$):**
   $$\min f_1(s) = \sum_{p=1}^{10} \sum_{i=1}^{n_p} c(s_{p,i}) \times 2.000$$
2. **$f_2$ (Desvio Quadrático da Sílica - $\%^2$):**
   $$\min f_2(s) = \sum_{p=1}^{10} \left(\overline{\text{SiO}_2}(p) - \text{alvo}_{\text{SiO}_2}^{\text{grupo}(p)}\right)^2$$
3. **$f_3$ (Desvio Quadrático da Alumina - $\%^2$):**
   $$\min f_3(s) = \sum_{p=1}^{10} \left(\overline{\text{Al}_2\text{O}_3}(p) - \text{alvo}_{\text{Al}_2\text{O}_3}^{\text{grupo}(p)}\right)^2$$

---

## 2. Evolução Metodológica da Etapa 1: Modelo Baseline v1 vs Modelo Canônico v2

### A Falha Estrutural do Modelo Baseline v1 (Penalidade Estática $W=10^7$)
Na formulação preliminar (v1), as restrições violadas eram tratadas somando uma penalidade linear estática $f(s) + 10^7 \times \text{violação}(s)$. Esse mecanismo clássico colapsou sob a disparidade de escalas:
1. **Falso Ótimo em $f_1$:** Uma pequena infração de qualidade (ex.: teor de sílica $0,014\%$ fora da faixa regulatória) custava apenas R$ 143 mil de penalidade, mas gerava uma economia de R$ 560 mil ao permitir o uso de minérios muito mais baratos. O algoritmo reportou um falso mínimo de R$ 60,88M que era **inviável**.
2. **Efeito Falésia em $f_2$ e $f_3$:** O peso de $10^7$ era ordens de grandeza maior que o valor dos objetivos quadráticos ($\approx 2,0$), destruindo os gradientes de qualidade da busca.
3. **Taxa de factibilidade de apenas 46,7% (7 em 15 execuções).**

### As Inovações do Modelo Canônico v2
1. **Princípio de Dominação de Restrições de Deb (2000):**
   - **Regra 1 (Prioridade Absoluta de Factibilidade):** Qualquer solução factível domina qualquer solução infactível, independente do custo aparente.
   - **Regra 2 (Entre Viáveis):** Menor valor puro do objetivo vence.
   - **Regra 3 (Entre Inviáveis):** Menor violação agregada de restrições ($\sum \text{disp} + \sum \text{qual}$) vence, guiando a busca rumo à fronteira factível.
   - **Benefício:** **Zero pesos de penalidade arbitrários.** 100% de factibilidade estrita garantida.
2. **Estrutura de Vizinhança Canônica $N_3$ (Substituição Tripla Distribuída):**
   - $N_1$: Substituição Simples em 1 pilha ($d_H = 1$).
   - $N_2$: Troca Cruzada entre 2 pilhas distintas ($d_H = 2$, 100% conservativa de estoque).
   - $N_3$: Substituição Tripla Distribuída em 3 pilhas distintas ($d_H = 3$). Altera simultaneamente 30% das pilhas operacionais da planta com 100% de elegibilidade garantida por construção, permitindo escapar de bacias locais sem violar capacidades.
3. **VND (Variable Neighborhood Descent) e GVNS Estocásticos Rápidos:**
   - Procedimento de busca local em cascata $N_1 \to N_2 \to N_3$ com *First-Improvement* estocástico e parada por estagnação (`max_sem_melhora`), permitindo dezenas de ciclos profundos de perturbação e busca local dentro do orçamento de avaliações.

---

## 3. Resultados Experimentais da Etapa 1 (100% Factíveis)

Execução realizada sobre 5 sementes estocásticas (`SEEDS = [10, 20, 30, 40, 50]`) com orçamento de $10.000$ avaliações:

| Objetivo | Runs | Factíveis | Taxa (%) | Mínimo Viável | Média Viável | Desvio Padrão | Solução Âncora ($s_i^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$f_1$ (Custo de Aquisição)** | 5 | 5 | **100%** | **R$ 60.840.000,00** | R$ 61.124.000,00 | R$ 214.252,18 | Seed 30 |
| **$f_2$ (Desvio $\text{SiO}_2$)** | 5 | 5 | **100%** | **2,4405 (%²)** | 2,4982 (%²) | 0,0512 (%²) | Seed 40 |
| **$f_3$ (Desvio $\text{Al}_2\text{O}_3$)** | 5 | 5 | **100%** | **1,4728 (%²)** | 1,5145 (%²) | 0,0284 (%²) | Seed 20 |

---

## 4. Fundamentos e Alicerces para a Etapa 2 (Teoria da Decisão Multiobjetivo)

### 4.1 Tabela de Payoff (Matriz de Conflito $3 \times 3$)
A Tabela de Payoff avalia as 3 soluções âncora ($s_1^*, s_2^*, s_3^*$) em todos os três objetivos:

| Solução Âncora | $f_1$ - Custo (R$) | $f_2$ - Desvio $\text{SiO}_2$ (%²) | $f_3$ - Desvio $\text{Al}_2\text{O}_3$ (%²) | Diagnóstico Operacional |
| :--- | :---: | :---: | :---: | :--- |
| **$s_1^*$ (Ótimo de Custo)** | **R$ 60.840.000,00** | 6,4973 | 5,4105 | Alocação dos minérios mais baratos (M17, M18, M20). |
| **$s_2^*$ (Ótimo de $\text{SiO}_2$)** | R$ 64.040.000,00 | **2,4405** | 3,2424 | Sintonia fina na sílica; acréscimo de +R$ 3,20M no custo. |
| **$s_3^*$ (Ótimo de $\text{Al}_2\text{O}_3$)** | R$ 65.600.000,00 | 4,8512 | **1,4728** | Sintonia fina na alumina; acréscimo de +R$ 4,76M no custo. |

### 4.2 Pontos de Referência Fundamentais
- **Ponto Ideal ($z^*$):** Mínimos absolutos atingíveis individualmente (diagonal da Payoff):
  $$z^* = (\text{R\$ } 60.840.000,00; \, 2,4405; \, 1,4728)$$
- **Ponto Utópico ($z^{*\text{uto}}$):** Ponto estritamente inatingível ($z^* - 0,01 \Delta$), garantindo estrita convexidade e unicidade de solução em Programação por Compromisso:
  $$z^{*\text{uto}} = (\text{R\$ } 60.792.400,00; \, 2,3999; \, 1,4334)$$
- **Ponto de Nadir ($z^{\text{nad}}$):** Pior valor de cada critério no conjunto de âncoras (limite superior da Frente de Pareto):
  $$z^{\text{nad}} = (\text{R\$ } 65.600.000,00; \, 6,4973; \, 5,4105)$$

### 4.3 Matriz de Conflito Normalizada (Deterioração Relativa %)
Mostra o quanto um objetivo se degrada ao priorizar exclusivamente outro critério:

| Decisão | Deterioração em $f_1$ | Deterioração em $f_2$ | Deterioração em $f_3$ | Conflito Observado |
| :--- | :---: | :---: | :---: | :--- |
| **Priorizar $f_1$ (Custo)** | **0,0%** (Ótimo) | **100,0%** (Pior) | **100,0%** (Pior) | Conflito severo com qualidade. |
| **Priorizar $f_2$ ($\text{SiO}_2$)** | 67,2% | **0,0%** (Ótimo) | 45,0% | Custo intermediário; conflito moderado com alumina. |
| **Priorizar $f_3$ ($\text{Al}_2\text{O}_3$)** | **100,0%** (Pior) | 59,4% | **0,0%** (Ótimo) | Custo mais alto da planta; conflito com sílica. |

### 4.4 Parâmetros de Normalização Min-Max para a Etapa 2
Para eliminar a disparidade de grandezas ($10^7$ em dinheiro vs $10^0$ em química):
$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*}$$
- **$f_1$:** Amplitude $\Delta_1 = \text{R\$ } 4.760.000,00 \implies \text{Fator de Escala } w_1 = 2,100840 \times 10^{-7}$
- **$f_2$:** Amplitude $\Delta_2 = 4,0568 \implies \text{Fator de Escala } w_2 = 0,246499$
- **$f_3$:** Amplitude $\Delta_3 = 3,9377 \implies \text{Fator de Escala } w_3 = 0,253955$

### 4.5 Malha Recomendada para o Método $\varepsilon$-Restrito (10 Passos)
Ao minimizar o custo $f_1(s)$ sujeito a $f_2(s) \le \varepsilon_2$ e $f_3(s) \le \varepsilon_3$:
- **$\varepsilon_2$ ($\text{SiO}_2$):** Intervalo `[2.4405, 6.4973]`
  - Passos: `[2.4405, 2.8913, 3.3420, 3.7928, 4.2435, 4.6943, 5.1450, 5.5958, 6.0465, 6.4973]`
- **$\varepsilon_3$ ($\text{Al}_2\text{O}_3$):** Intervalo `[1.4728, 5.4105]`
  - Passos: `[1.4728, 1.9103, 2.3478, 2.7854, 3.2229, 3.6604, 4.0979, 4.5355, 4.9730, 5.4105]`

---

## 5. Estrutura do Repositório

```
TP1TD/
├── data.py                 # Definições de minérios, pilhas, grupos e teores alvos
├── model.py                # Funções objetivo, checagem de restrições e Regras de Deb
├── construction.py         # Heurística construtiva inicial
├── neighborhoods.py        # Vizinhanças N1, N2 e N3 (Shaking e Descida VND)
├── vns.py                  # Implementação canônica do VND e GVNS
├── experiments.py          # Pipeline de experimentos, estatísticas, payoff e gráficos
├── run_tests.py            # Executor da suíte completa de testes unitários
├── tests/                  # 48 testes automatizados com diagnósticos explícitos
│   ├── test_data.py
│   ├── test_construction.py
│   ├── test_feasibility.py
│   ├── test_objectives.py
│   ├── test_penalty_and_modeling.py
│   ├── test_neighborhoods.py
│   ├── test_vns_mechanics.py
│   └── test_decision_theory.py
├── outputs/                # Artefatos da execução mais recente (gráficos, JSON, relatórios)
├── v1outs/                 # Backup histórico dos experimentos do baseline v1
└── v2outs/                 # Backup consolidado dos experimentos do modelo canônico v2
```

---

## 6. Instruções de Execução e Reprodução

### Instalação de Dependências
Requer Python 3.9+ e os pacotes matemáticos padrão:
```bash
python3 -m pip install numpy matplotlib
```

### Execução da Suíte de Testes (48 Testes com Diagnósticos Explícitos)
```bash
python3 run_tests.py
# ou
python3 -m unittest discover tests -v
```

### Reprodução Completa dos Experimentos da Etapa 1
Executa as 15 otimizações mono-objetivo, gera a Matriz de Payoff, calcula os pontos de referência e exporta os relatórios:
```bash
python3 experiments.py
```

Os artefatos gráficos gerados incluem:
- `convergencia_f1.png`, `convergencia_f2.png`, `convergencia_f3.png` (painéis duplos com fase global e zoom fino factível)
- `matriz_conflito_payoff.png` (projeções 2D do espaço de critérios e heatmap de conflito)
- `comparativo_v1_vs_v2.png` (taxa de viabilidade e eliminação de falsos ótimos)
- `melhor_solucao_f1.png`, `melhor_solucao_f2.png`, `melhor_solucao_f3.png` (teores por pilha vs limites regulatórios)
- `proximidade_alvos_detalhada.png` (desvios absolutos pilha a pilha)
- `uso_minerios_v2.png` (saturação logística do pátio de minérios)
- `relatorio_entrega1_v2.md` e `resultados_entrega1_v2.json`
