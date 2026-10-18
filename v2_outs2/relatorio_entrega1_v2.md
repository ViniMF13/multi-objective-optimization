# Relatório Técnico - Entrega 1 & Preparação para a Entrega 2
## Otimização Mono-Objetivo Canônica (GVNS + Deb + N3) e Fundamentos de Teoria da Decisão Multiobjetivo

> **Resumo Executivo:**
> Este relatório apresenta os resultados definitivos da Entrega 1, incorporando o **Princípio de Dominação de Restrições de Deb (2000)** > e a nova estrutura de vizinhança **N3 (Substituição Tripla Distribuída)**. > Com a eliminação completa dos falsos ótimos decorrentes da penalidade estática, obtivemos **100% de factibilidade estrita** em todos os testes. > Adicionalmente, estruturamos os alicerces teóricos para a **Entrega 2**: a **Matriz de Payoff (Conflito)**, o **Ponto Ideal ($z^*$)**, > o **Ponto Utópico ($z^{*\text{uto}}$)**, o **Ponto de Nadir ($z^{\text{nad}}$)** e as funções de normalização para métodos multiobjetivo.

## 1. Estatísticas Descritivas das Execuções Mono-Objetivo (5 Sementes)

| Objetivo | Runs | Viáveis | Taxa (%) | Mínimo Puro | Média Pura | Desvio Padrão | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **f1** (Custo) | 5 | 5 | **100%** | 60,840,000.0000 | 61,096,000.0000 | 192,831.5327 | **100% Factível** |
| **f2** (SiO2) | 5 | 5 | **100%** | 2.4405 | 2.5093 | 0.0710 | **100% Factível** |
| **f3** (Al2O3) | 5 | 5 | **100%** | 1.4728 | 1.5097 | 0.0326 | **100% Factível** |

## 2. Matriz de Payoff (Tabela de Pagamentos / Matriz de Conflito 3x3)

A Matriz de Payoff avalia cada uma das 3 soluções âncora ($s_1^*, s_2^*, s_3^*$) em todas as três funções objetivo, evidenciando os custos de oportunidade e o conflito inerente entre os critérios:

| Solução Âncora | f1 - Custo (R$) | f2 - Desvio SiO2 (%²) | f3 - Desvio Al2O3 (%²) | Descrição Operacional |
| :--- | :---: | :---: | :---: | :--- |
| **s1* (Ótimo Custo)** | **R$ 60,840,000.00** | 6.4973 | 5.4105 | Prioriza minérios mais baratos (M17, M18, M20) |
| **s2* (Ótimo SiO2)** | R$ 63,740,000.00 | **2.4405** | 3.4824 | Sintoniza com precisão o teor de sílica |
| **s3* (Ótimo Al2O3)** | R$ 65,600,000.00 | 5.4454 | **1.4728** | Sintoniza com precisão o teor de alumina |

## 3. Pontos de Referência de Teoria da Decisão

| Ponto de Referência | Vetor no Espaço de Critérios $(f_1, f_2, f_3)$ | Papel Teórico e Operacional |
| :--- | :---: | :--- |
| **Ponto Ideal ($z^*$)** | `(R$ 60,840,000.00, 2.4405, 1.4728)` | Vetor dos mínimos absolutos atingíveis individualmente (diagonal da Payoff). |
| **Ponto Utópico ($z^{*\text{uto}}$)** | `(R$ 60,792,400.00, 2.4000, 1.4334)` | Ponto estritamente inatingível ($z^* - 0,01 \Delta$), usado para garantir unicidade em Tchebycheff. |
| **Ponto de Nadir ($z^{\text{nad}}$)** | `(R$ 65,600,000.00, 6.4973, 5.4105)` | Vetor dos piores valores obtidos no conjunto de âncoras (teto do Espaço de Pareto). |

## 4. Matriz de Conflito Normalizada (Degradação Percentual)

Mostra o grau de deterioração percentual $\Delta_{ij} = \frac{P_{ij} - z_j^*}{z_j^{\text{nad}} - z_j^*} \times 100\%$ ao adotar uma solução âncora sobre os demais objetivos:

| Solução Adotada | Deterioração em Custo (f1) | Deterioração em SiO2 (f2) | Deterioração em Al2O3 (f3) | Nível de Conflito |
| :--- | :---: | :---: | :---: | :--- |
| Otimizando exclusivamente **f1 (Custo)** | **0.0%** (Ótimo) | 100.0% | 100.0% | Conflito severo com a qualidade |
| Otimizando exclusivamente **f2 (SiO2)** | 60.9% | **0.0%** (Ótimo) | 51.0% | Custo moderado, conflito com alumina |
| Otimizando exclusivamente **f3 (Al2O3)** | 100.0% | 74.1% | **0.0%** (Ótimo) | Custo moderado, conflito com sílica |

## 5. Preparação e Parâmetros para a Entrega 2 (Otimização Multiobjetivo)

### 5.1 Fatores de Normalização Min-Max
Para equalizar a disparidade de escalas ($10^7$ em custo financeiro vs $10^0$ em desvios químicos), a função de normalização é dada por:
$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*}$$

| Objetivo | Ponto Ideal ($z_i^*$) | Ponto Nadir ($z_i^{\text{nad}}$) | Amplitude ($\Delta_i$) | Fator de Escala ($1 / \Delta_i$) |
| :--- | :---: | :---: | :---: | :---: |
| **f1** (Custo) | 60,840,000.0000 | 65,600,000.0000 | 4,760,000.0000 | `2.100840e-07` |
| **f2** (SiO2) | 2.4405 | 6.4973 | 4.0567 | `2.465028e-01` |
| **f3** (Al2O3) | 1.4728 | 5.4105 | 3.9377 | `2.539573e-01` |

### 5.2 Malha Sugerida para o Método $\varepsilon$-Restrito (10 Intervalos)
Ao minimizar o custo $f_1(s)$ sujeito a $f_2(s) \le \varepsilon_2$ e $f_3(s) \le \varepsilon_3$, os limites operacionais recomendados para os valores de $\varepsilon$ são:

- **Tolerância de SiO2 (eps2):** Intervalo `[2.4405, 6.4973]` (%²)
  - Malha discretizada (10 pontos): `[2.4405, 2.8913, 3.342, 3.7928, 4.2435, 4.6943, 5.145, 5.5958, 6.0465, 6.4973]`
- **Tolerância de Al2O3 (eps3):** Intervalo `[1.4728, 5.4105]` (%²)
  - Malha discretizada (10 pontos): `[1.4728, 1.9103, 2.3478, 2.7854, 3.2229, 3.6604, 4.0979, 4.5354, 4.973, 5.4105]`

## 6. Comparativo Metodológico: Baseline v1 vs Modelo v2

| Métrica de Avaliação | Modelo Baseline v1 (Penalidade Estática $10^7$) | Modelo Canônico v2 (Deb + Vizinhança N3) | Impacto Teórico / Prático |
| :--- | :---: | :---: | :--- |
| **Taxa de Viabilidade Geral** | 46.7% (7/15 execuções) | **100.0% (15/15 execuções)** | Erradicação total de soluções infactíveis. |
| **Taxa de Viabilidade em f1** | 60.0% (3/5 execuções) | **100.0% (5/5 execuções)** | Eliminação do falso mínimo inviável. |
| **Melhor Custo Factível (f1)** | R$ 61.440.000,00 | **R$ 60,840,000.00** | Garantia de ótimo estritamente viável. |
| **Sensibilidade a Hiperparâmetros** | Extrema ($W=10^7$ arbitrário) | **Nula (Zero pesos de penalidade)** | Modelo auto-suficiente e robusto. |
| **Estrutura de Vizinhança N3** | Dupla troca local | **Substituição Tripla Distribuída (3 pilhas)** | Escape eficaz de bacias locais sem violações. |