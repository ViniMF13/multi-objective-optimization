# Case UFMG 2026/2 - Teoria da Decisão
## Otimização de Custo e Qualidade de Pilhas de Minério de Ferro para Sinterização

Este repositório contém a formulação matemática, algoritmos de metaheurística, rotinas de escalarização multiobjetivo e análises experimentais completas para as **Entregas #1 (Mono-Objetivo)** e **#2 (Multiobjetivo)**, estabelecendo as bases computacionais para a **Entrega #3 (Tomada de Decisão Multicritério - MCDA)**.

---

## 1. Descrição do Problema e Modelagem Matemática

O problema consiste em determinar a composição ótima de **10 pilhas de mistura/homogeneização de minério de ferro** (totalizando 290.000 toneladas / 145 caminhões de 2 kt cada) que alimentam dois sinterizadores industriais (**Sinter 1** e **Sinter 2**):

- **Sinter 1 (Pilhas P1 a P5):** 60 caminhões (120 kt). Limites químicos regulatórios: $\text{SiO}_2 \in [5,0; 6,2]\%$ (alvo operacional: $5,55\%$), $\text{Al}_2\text{O}_3 \in [1,0; 2,3]\%$ (alvo operacional: $1,25\%$).
- **Sinter 2 (Pilhas P6 a P10):** 85 caminhões (170 kt). Limites químicos regulatórios: $\text{SiO}_2 \in [5,0; 6,5]\%$ (alvo operacional: $5,50\%$), $\text{Al}_2\text{O}_3 \in [1,0; 2,2]\%$ (alvo operacional: $1,50\%$).
- **Disponibilidade e Logística:** 20 tipos de minérios (M1 a M20), cada um com limites de disponibilidade mínima ($\text{disp\_min}$) e máxima ($\text{disp\_max}$), regras estritas de elegibilidade tecnológica de pátio (S1 e/ou S2) e custo unitário de aquisição (R$/t).

### 1.1 Funções Objetivo
1. **$f_1$ (Custo Financeiro Total de Aquisição - R$):**
   $$\min f_1(s) = \sum_{p=1}^{10} \sum_{i=1}^{n_p} c(s_{p,i}) \times 2.000$$
2. **$f_2$ (Desvio Quadrático Médio da Sílica - $\%^2$):**
   $$\min f_2(s) = \sum_{p=1}^{10} \left(\overline{\text{SiO}_2}(p) - \text{alvo}_{\text{SiO}_2}^{\text{grupo}(p)}\right)^2$$
3. **$f_3$ (Desvio Quadrático Médio da Alumina - $\%^2$):**
   $$\min f_3(s) = \sum_{p=1}^{10} \left(\overline{\text{Al}_2\text{O}_3}(p) - \text{alvo}_{\text{Al}_2\text{O}_3}^{\text{grupo}(p)}\right)^2$$

### 1.2 Restrições do Sistema
- Atendimento exato da massa de cada pilha (garantido por construção com vetores de comprimento fixo).
- Respeito estrito à elegibilidade de cada minério para o grupo correspondente.
- Respeito às disponibilidades mínima e máxima de cada minério ($\sum \text{uso}_m$).
- Respeito às faixas regulatórias de teor de $\text{SiO}_2$ e $\text{Al}_2\text{O}_3$ em todas as 10 pilhas.

---

## 2. Síntese da Etapa 1: Motor de Busca Mono-Objetivo Canônico

Para superar a falha do modelo preliminar v1 (que utilizava penalidade linear estática $W=10^7$ e gerava falsos ótimos inviáveis em $f_1$), foi adotada uma arquitetura canônica e rigorosa:

1. **Princípio de Dominação de Restrições de Deb (2000):**
   - **Regra 1 (Factibilidade Absoluta):** Qualquer solução factível domina qualquer solução infactível, independente do custo aparente.
   - **Regra 2 (Entre Viáveis):** Menor valor puro do objetivo vence.
   - **Regra 3 (Entre Inviáveis):** Menor violação agregada de restrições vence, guiando a busca em direção à fronteira viável.
   - **Vantagem:** Eliminação total de parâmetros arbitrários de penalização; 100% de factibilidade estrita garantida.
2. **Estruturas de Vizinhança Canônicas ($N_1, N_2, N_3$):**
   - **$N_1$ (Substituição Simples):** Sorteia 1 pilha e altera 1 caminhão por outro minério elegível ($d_H = 1$).
   - **$N_2$ (Troca Entre Pilhas Distintas):** Troca 2 caminhões mutuamente compatíveis entre pilhas distintas ($d_H = 2$, 100% conservativo de estoque global).
   - **$N_3$ (Substituição Tripla Distribuída):** Altera simultaneamente 1 caminhão em 3 pilhas distintas ($d_H = 3$), afetando 30% da planta de forma balanceada e com 100% de elegibilidade por construção.
3. **Metaheurística GVNS + VND:**
   - Procedimento de busca local VND em cascata determinística ($N_1 \to N_2 \to N_3$) com reinício em $N_1$ a cada melhora e critério de parada por estagnação (`max_sem_melhora`), acoplado a macro-ciclos de perturbação estocástica (Shaking).

### Resultados Âncora e Pontos de Referência da Etapa 1

| Ponto de Referência | Custo $f_1$ (R$) | Desvio $\text{SiO}_2$ ($f_2$) | Desvio $\text{Al}_2\text{O}_3$ ($f_3$) | Interpretação Teórica |
| :--- | :---: | :---: | :---: | :--- |
| **Ótimo de Custo ($s_1^*$)** | **R$ 60.840.000,00** | 6,4973 | 5,4105 | Minérios mais baratos da planta; pior qualidade química. |
| **Ótimo de $\text{SiO}_2$ ($s_2^*$)** | R$ 64.040.000,00 | **2,4405** | 3,2424 | Alvo de sílica atingido; acréscimo de +R$ 3,20M no custo. |
| **Ótimo de $\text{Al}_2\text{O}_3$ ($s_3^*$)** | R$ 65.600.000,00 | 4,8512 | **1,4728** | Alvo de alumina atingido; acréscimo de +R$ 4,76M no custo. |
| **Ponto Ideal ($z^*$)** | **R$ 60.840.000,00** | **2,4405** | **1,4728** | Mínimos absolutos (diagonal da Matriz de Payoff). |
| **Ponto Nadir ($z^{\text{nad}}$)** | **R$ 65.600.000,00** | **6,4973** | **5,4105** | Piores valores da Payoff (fronteira superior de Pareto). |
| **Ponto Utópico ($z^{*\text{uto}}$)** | **R$ 60.792.400,00** | **2,4000** | **1,4334** | Ponto estritamente inatingível ($z^* - 0,01\Delta$). |
| **Amplitude ($\Delta_i$)** | **R$ 4.760.000,00** | **4,0567** | **3,9377** | Intervalo utilizado na Normalização Min-Max. |

---

## 3. Entrega #2: Otimização Multiobjetivo (MOP)

A segunda etapa do trabalho expande a otimização mono-objetivo para o espaço vetorial tridimensional $\mathbf{F}(s) = [f_1(s), f_2(s), f_3(s)]^T$, utilizando o mesmo núcleo computacional (GVNS + VND + Deb) como resolvedor escalar subjacente para mapear a **Fronteira de Pareto**.

### 3.1 Formulação Matemática das Abordagens Escalares

#### 1. Método da Soma Ponderada ($P_w$)
Agrega os três objetivos normalizados através de combinações convexas no simplex unitário:

$$\min_{s \in \mathcal{S}} \quad P_w(s; \boldsymbol{\lambda}) = \lambda_1 f_1^{\text{norm}}(s) + \lambda_2 f_2^{\text{norm}}(s) + \lambda_3 f_3^{\text{norm}}(s)$$

$$\text{sujeito a } \sum_{i=1}^3 \lambda_i = 1, \quad \lambda_i \ge 0, \quad s \in \mathcal{S}$$

Onde a **Normalização Min-Max** coloca todos os critérios na escala adimensional $[0, 1]$:
$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*}$$

- **Grade de Pesos:** Simplex-Lattice Design com passo $\Delta \lambda = 0,25$, gerando **15 vetores de pesos** uniformemente distribuídos entre vértices, arestas e centroides da 3-simplex.

#### 2. Método do $\varepsilon$-Restrito ($P_\varepsilon$)
Mantém o objetivo financeiro primário na função objetivo e transforma os desvios químicos em restrições com cotas superiores $\varepsilon_2, \varepsilon_3$:

$$\min_{s \in \mathcal{S}} \quad f_1(s)$$

$$\text{sujeito a } f_2(s) \le \varepsilon_2, \quad f_3(s) \le \varepsilon_3, \quad s \in \mathcal{S}$$

- **Integração Nativa com as Regras de Deb:** Os cortes de qualidade são adicionados diretamente ao vetor de violação de restrições de Deb, normalizados pelas respectivas amplitudes:
  $$v_2(s) = \frac{\max(0, \; f_2(s) - \varepsilon_2)}{\Delta_2}, \quad v_3(s) = \frac{\max(0, \; f_3(s) - \varepsilon_3)}{\Delta_3}$$
  $$\text{Violação Total}_\varepsilon(s) = \text{Violação Física}(s) + v_2(s) + v_3(s)$$
  Garantindo que uma solução que respeite os limites $\varepsilon$ domine qualquer solução que os viole, sem necessidade de sintonia de pesos de penalidade.
- **Grade Bidimensional de $\varepsilon$:** Grade $4 \times 4$ discretizada no intervalo $[z_k^*, z_k^{\text{nad}}]$, gerando **16 pares de cotas** $(\varepsilon_2, \varepsilon_3)$.

---

### 3.2 Protocolo Experimental Multiobjetivo
- **Sementes Estocásticas:** 5 execuções (`SEEDS = [10, 20, 30, 40, 50]`) para cada configuração.
- **Total de Execuções:**
  - Soma Ponderada: 15 configurações $\times$ 5 seeds = **75 execuções**.
  - $\varepsilon$-Restrito: 16 pares de corte $\times$ 5 seeds = **80 execuções**.
  - Total geral: **155 problemas escalares resolvidos** via GVNS + VND.
- **Orçamento Computacional:** 1.500 avaliações por subproblema (`ls_evals_limit=120`), concluído em **menos de 1 minuto** total no ambiente local.

---

### 3.3 Resultados Experimentais Consolidados e Métricas de Qualidade

| Métrica de Desempenho | Soma Ponderada ($P_w$) | $\varepsilon$-Restrito ($P_\varepsilon$) | Fronteira Combinada Global ($\mathcal{PF}_{\text{global}}$) |
| :--- | :---: | :---: | :---: |
| **Configurações Avaliadas** | 15 vetores de pesos | 16 pares de cortes | 31 subproblemas |
| **Total de Execuções (5 seeds)** | 75 | 80 | 155 |
| **Taxa de Viabilidade Geral** | **93,3%** (70/75) | **50,0%** (40/80)* | — |
| **Soluções Não-Dominadas ($|\mathcal{PF}|$)** | **49** | **21** | **58** |
| **Hipervolume 3D ($HV$)** | **0,6139** | **0,4216** | **0,6170** |
| **Espaçamento de Schott ($S$)** | **0,0458** | **0,0532** | **0,0407** |
| **Contribuição na Frente Global** | 82,8% (48/58) | 17,2% (10/58) | 100,0% |
| **Métrica de Cobertura $C(A, B)$** | $C(P_w, P_\varepsilon) = 52,38\%$ | $C(P_\varepsilon, P_w) = 2,04\%$ | — |

*\*Nota: A viabilidade de 50,0% no $\varepsilon$-restrito ocorre porque as combinações em que tanto $\varepsilon_2$ quanto $\varepsilon_3$ são fixados simultaneamente próximos aos seus valores ideais ($z^*$) são fisicamente inalcançáveis devido ao conflito severo entre sílica e alumina. O algoritmo tratou com exatidão essas instâncias como infactíveis via Regras de Deb.*

---

### 3.4 Análise Crítica: $P_w$ vs. $P_\varepsilon$ na Teoria da Otimização

1. **Incapacidade da Soma Ponderada em Regiões Não-Convexas:**
   - Como previsto pela literatura clássica (*Miettinen, 1999; Ehrgott, 2005*), a soma ponderada é incapaz de capturar soluções eficientes localizadas em concavidades da fronteira de Pareto (*non-supported efficient solutions*).
   - O método $\varepsilon$-restrito, por sua vez, demonstrou capacidade de penetrar nessas regiões ao impor limites rígidos de qualidade, contribuindo com **10 soluções exclusivas** que não puderam ser encontradas pela combinação linear de pesos.
2. **Hipervolume e Densidade da Frente:**
   - A combinação dos dois métodos gerou a melhor fronteira agregada ($HV = 0,6170$ e menor espaçamento $S = 0,0407$), demonstrando que as duas técnicas são complementares no mapeamento do espaço de critérios.
3. **Variabilidade entre Sementes:**
   - As 5 sementes estocásticas convergiram para regiões adjacentes e consistentes do espaço tridimensional. O arquivo de não-dominadas refinou continuamente a fronteira ao reter a envoltória inferior de todas as execuções.

---

## 4. Visualizações e Artefatos Gráficos Multiobjetivo

Os gráficos gerados estão disponíveis no diretório `outputs/`:

1. **Projeções 2D da Soma Ponderada (`outputs/projecoes_soma_ponderada.png`):**
   - Três planos ortogonais ($f_1 \times f_2$, $f_1 \times f_3$, $f_2 \times f_3$) exibindo os pontos de cada uma das 5 sementes e destacando a fronteira de Pareto com 49 soluções.
2. **Projeções 2D do $\varepsilon$-Restrito (`outputs/projecoes_epsilon_restrito.png`):**
   - Três planos ortogonais com as 5 sementes e as 21 soluções da fronteira de corte $\varepsilon$.
3. **Comparativo Sobreposto das Abordagens (`outputs/comparativo_pareto_pw_vs_peps.png`):**
   - Sobreposição direta de $P_w$ (círculos azuis), $P_\varepsilon$ (quadrados laranjas) e Fronteira Global Não-Dominada (estrelas vermelhas), evidenciando o preenchimento de vazios pelo método $\varepsilon$-restrito.
4. **Fronteira de Pareto Tridimensional (`outputs/fronteira_pareto_3d.png`):**
   - Visualização 3D completa no espaço de critérios $[f_1, f_2, f_3]$ com os pontos Ideal $z^*$ e Nadir $z^{\text{nad}}$.

---

## 5. Estrutura do Repositório

```
TP1TD/
├── data.py                       # Definições de minérios, pilhas, grupos e teores operacionais
├── model.py                      # Funções objetivo (f1, f2, f3), restrições físicas e Regras de Deb
├── construction.py               # Heurística construtiva gulosa baseada em custo
├── neighborhoods.py              # Operadores N1, N2 e N3 (Shaking estocástico e Descida VND)
├── vns.py                        # Implementação canônica do VND e GVNS com suporte escalar
├── multiobjective.py             # Escalarização (Pw, Peps), dominância, hipervolume e espaçamento
├── experiments.py                # Pipeline experimental da Etapa 1 (Mono-objetivo, Payoff e Conflito)
├── experiments_mop.py            # Pipeline experimental da Etapa 2 (Multiobjetivo Pw e Peps, 155 runs)
├── run_tests.py                  # Executor da suíte de testes com diagnóstico explícito
├── tests/                        # 56 testes automatizados com traces detalhados
│   ├── test_data.py              # Validação de integridade dos dados e parâmetros
│   ├── test_construction.py      # Testes da heurística construtiva inicial
│   ├── test_feasibility.py       # Validação das restrições de massa, elegibilidade e pátio
│   ├── test_objectives.py        # Testes das funções objetivo f1, f2 e f3
│   ├── test_penalty_and_modeling.py # Testes rigorosos das Regras de Dominação de Deb
│   ├── test_neighborhoods.py     # Auditoria dos operadores N1, N2 e N3 com Hamming distance
│   ├── test_vns_mechanics.py     # Validação da cascata VND, Shaking e reprodutibilidade do GVNS
│   ├── test_decision_theory.py   # Testes da Tabela de Payoff, pontos z*, z^nad e z^uto
│   └── test_multiobjective.py    # Testes das classes escalares, dominância e métricas (HV e S)
├── outputs/                      # Artefatos da execução mais recente (gráficos, JSON e relatórios)
│   ├── resultados_entrega1_v2.json  # Payoff matrix e pontos de referência da Etapa 1
│   ├── resultados_entrega2.json     # Consolidação completa dos 155 subproblemas multiobjetivo
│   ├── solucoes_pareto_etapa2.json  # Base de dados das 58 soluções não-dominadas para Etapa 3
│   ├── relatorio_entrega1_v2.md     # Relatório analítico da Etapa 1
│   ├── relatorio_entrega2.md        # Relatório analítico da Etapa 2
│   ├── comparativo_pareto_pw_vs_peps.png
│   ├── fronteira_pareto_3d.png
│   ├── projecoes_soma_ponderada.png
│   └── projecoes_epsilon_restrito.png
├── v1outs/                       # Backup histórico dos experimentos do baseline v1
└── v2outs/                       # Backup consolidado dos experimentos do modelo canônico v2
```

---

## 6. Instruções de Execução e Reprodução

### Ambiente e Dependências
Requer Python 3.9+ e os pacotes científicos padrão:
```bash
python3 -m pip install numpy matplotlib
```

### 1. Execução da Suíte de Testes (56 Testes Unitários e de Integração)
Todos os testes foram implementados com diagnósticos explícitos (exibindo valores de entrada, deltas, violações e comprovações matemáticas):
```bash
python3 run_tests.py
```

### 2. Reprodução da Etapa 1 (Mono-Objetivo)
Executa as 15 otimizações mono-objetivo, gera a Matriz de Payoff e os pontos de referência:
```bash
python3 experiments.py
```

### 3. Reprodução da Etapa 2 (Otimização Multiobjetivo)
Executa os 155 subproblemas escalares ($P_w$ e $P_\varepsilon$ através das 5 sementes), filtra as soluções não dominadas, calcula Hipervolume e Espaçamento e gera os gráficos 2D e 3D:
```bash
python3 experiments_mop.py --evals 1500 --ls-evals 120
```

Para regenerar apenas as figuras e relatórios a partir dos dados já salvos em `outputs/resultados_entrega2.json`:
```bash
python3 experiments_mop.py --replot
```

---

## 7. Transição para a Entrega #3 (Tomada de Decisão Multicritério - MCDA)

O arquivo `outputs/solucoes_pareto_etapa2.json` exporta as **58 soluções eficientes** da Fronteira de Pareto com todas as informações necessárias para a Etapa 3:
1. **Composição completa das 10 pilhas** (145 caminhões alocados).
2. **Valores dos três atributos originais:** Custo total ($f_1$), Desvio de Sílica ($f_2$) e Desvio de Alumina ($f_3$).
3. **Cálculo do Quarto Atributo de Decisão:** Homogeneidade de receitas entre pilhas do mesmo equipamento (desvio médio em relação ao vetor médio de minérios do grupo).
4. **Aplicação dos Métodos de Decisão:** Utilização de métodos de sobreclassificação (**ELECTRE / PROMETHEE**) ou agregação hierárquica (**AHP**) para recomendar a alternativa de compromisso ótima para a operação da usina siderúrgica.
