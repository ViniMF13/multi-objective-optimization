# Relatório Técnico - Entrega 1 & Preparação para a Entrega 2
## Otimização Mono-Objetivo Canônica (GVNS + Deb + N3) e Fundamentos de Teoria da Decisão Multiobjetivo

> **Resumo Executivo:**
> Este relatório apresenta os resultados definitivos da Entrega 1, incorporando o **Princípio de Dominação de Restrições de Deb (2000)** > e a nova estrutura de vizinhança **N3 (Substituição Tripla Distribuída)**. > Com a eliminação completa dos falsos ótimos decorrentes da penalidade estática, obtivemos **100% de factibilidade estrita** em todos os testes. > Adicionalmente, estruturamos os alicerces teóricos para a **Entrega 2**: a **Matriz de Payoff (Conflito)**, o **Ponto Ideal ($z^*$)**, > o **Ponto Utópico ($z^{*\text{uto}}$)**, o **Ponto de Nadir ($z^{\text{nad}}$)** e as funções de normalização para métodos multiobjetivo.

## 1. Estatísticas Descritivas das Execuções Mono-Objetivo (5 Sementes)

| Objetivo | Runs | Viáveis | Taxa (%) | Mínimo Puro | Média Pura | Desvio Padrão | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **f1** (Custo) | 5 | 5 | **100%** | 61,140,000.0000 | 61,384,000.0000 | 174,539.3938 | **100% Factível** |
| **f2** (SiO2) | 5 | 5 | **100%** | 2.5075 | 2.5537 | 0.0653 | **100% Factível** |
| **f3** (Al2O3) | 5 | 5 | **100%** | 1.5091 | 1.5394 | 0.0310 | **100% Factível** |

## 2. Matriz de Payoff (Tabela de Pagamentos / Matriz de Conflito 3x3)

A Matriz de Payoff avalia cada uma das 3 soluções âncora ($s_1^*, s_2^*, s_3^*$) em todas as três funções objetivo, evidenciando os custos de oportunidade e o conflito inerente entre os critérios:

| Solução Âncora | f1 - Custo (R$) | f2 - Desvio SiO2 (%²) | f3 - Desvio Al2O3 (%²) | Descrição Operacional |
| :--- | :---: | :---: | :---: | :--- |
| **s1* (Ótimo Custo)** | **R$ 61,140,000.00** | 6.0632 | 5.0799 | Prioriza minérios mais baratos (M17, M18, M20) |
| **s2* (Ótimo SiO2)** | R$ 64,040,000.00 | **2.5075** | 3.2424 | Sintoniza com precisão o teor de sílica |
| **s3* (Ótimo Al2O3)** | R$ 65,600,000.00 | 4.8512 | **1.5091** | Sintoniza com precisão o teor de alumina |

## 3. Pontos de Referência de Teoria da Decisão

| Ponto de Referência | Vetor no Espaço de Critérios $(f_1, f_2, f_3)$ | Papel Teórico e Operacional |
| :--- | :---: | :--- |
| **Ponto Ideal ($z^*$)** | `(R$ 61,140,000.00, 2.5075, 1.5091)` | Vetor dos mínimos absolutos atingíveis individualmente (diagonal da Payoff). |
| **Ponto Utópico ($z^{*\text{uto}}$)** | `(R$ 61,095,400.00, 2.4719, 1.4734)` | Ponto estritamente inatingível ($z^* - 0,01 \Delta$), usado para garantir unicidade em Tchebycheff. |
| **Ponto de Nadir ($z^{\text{nad}}$)** | `(R$ 65,600,000.00, 6.0632, 5.0799)` | Vetor dos piores valores obtidos no conjunto de âncoras (teto do Espaço de Pareto). |

## 4. Matriz de Conflito Normalizada (Degradação Percentual)

Mostra o grau de deterioração percentual $\Delta_{ij} = \frac{P_{ij} - z_j^*}{z_j^{\text{nad}} - z_j^*} \times 100\%$ ao adotar uma solução âncora sobre os demais objetivos:

| Solução Adotada | Deterioração em Custo (f1) | Deterioração em SiO2 (f2) | Deterioração em Al2O3 (f3) | Nível de Conflito |
| :--- | :---: | :---: | :---: | :--- |
| Otimizando exclusivamente **f1 (Custo)** | **0.0%** (Ótimo) | 100.0% | 100.0% | Conflito severo com a qualidade |
| Otimizando exclusivamente **f2 (SiO2)** | 65.0% | **0.0%** (Ótimo) | 48.5% | Custo moderado, conflito com alumina |
| Otimizando exclusivamente **f3 (Al2O3)** | 100.0% | 65.9% | **0.0%** (Ótimo) | Custo moderado, conflito com sílica |

## 5. Preparação e Parâmetros para a Entrega 2 (Otimização Multiobjetivo)

### 5.1 Fatores de Normalização Min-Max
Para equalizar a disparidade de escalas ($10^7$ em custo financeiro vs $10^0$ em desvios químicos), a função de normalização é dada por:
$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*}$$

| Objetivo | Ponto Ideal ($z_i^*$) | Ponto Nadir ($z_i^{\text{nad}}$) | Amplitude ($\Delta_i$) | Fator de Escala ($1 / \Delta_i$) |
| :--- | :---: | :---: | :---: | :---: |
| **f1** (Custo) | 61,140,000.0000 | 65,600,000.0000 | 4,460,000.0000 | `2.242152e-07` |
| **f2** (SiO2) | 2.5075 | 6.0632 | 3.5557 | `2.812347e-01` |
| **f3** (Al2O3) | 1.5091 | 5.0799 | 3.5709 | `2.800444e-01` |

### 5.2 Malha Sugerida para o Método $\varepsilon$-Restrito (10 Intervalos)
Ao minimizar o custo $f_1(s)$ sujeito a $f_2(s) \le \varepsilon_2$ e $f_3(s) \le \varepsilon_3$, os limites operacionais recomendados para os valores de $\varepsilon$ são:

- **Tolerância de SiO2 (eps2):** Intervalo `[2.5075, 6.0632]` (%²)
  - Malha discretizada (10 pontos): `[2.5075, 2.9026, 3.2977, 3.6927, 4.0878, 4.4829, 4.878, 5.2731, 5.6681, 6.0632]`
- **Tolerância de Al2O3 (eps3):** Intervalo `[1.5091, 5.0799]` (%²)
  - Malha discretizada (10 pontos): `[1.5091, 1.9058, 2.3026, 2.6994, 3.0961, 3.4929, 3.8896, 4.2864, 4.6832, 5.0799]`

## 6. Comparativo Metodológico: Baseline v1 vs Modelo v2

| Métrica de Avaliação | Modelo Baseline v1 (Penalidade Estática $10^7$) | Modelo Canônico v2 (Deb + Vizinhança N3) | Impacto Teórico / Prático |
| :--- | :---: | :---: | :--- |
| **Taxa de Viabilidade Geral** | 46.7% (7/15 execuções) | **100.0% (15/15 execuções)** | Erradicação total de soluções infactíveis. |
| **Taxa de Viabilidade em f1** | 60.0% (3/5 execuções) | **100.0% (5/5 execuções)** | Eliminação do falso mínimo inviável. |
| **Melhor Custo Factível (f1)** | R$ 61.440.000,00 | **R$ 61,140,000.00** | Garantia de ótimo estritamente viável. |
| **Sensibilidade a Hiperparâmetros** | Extrema ($W=10^7$ arbitrário) | **Nula (Zero pesos de penalidade)** | Modelo auto-suficiente e robusto. |
| **Estrutura de Vizinhança N3** | Dupla troca local | **Substituição Tripla Distribuída (3 pilhas)** | Escape eficaz de bacias locais sem violações. |