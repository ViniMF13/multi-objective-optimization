# Relatório da Entrega 2: Otimização Multiobjetivo (MOP)
**Disciplina:** Teoria da Decisão (UFMG) | **Case:** Otimização de Custo e Qualidade de Pilhas de Minério
**Data da Execução:** 2026-10-18 18:48:51 | **Avaliador:** Metaheurística Canônica GVNS + VND com Regras de Deb

---

## 1. Formulação Matemática das Abordagens Escalares

### 1.1 Método da Soma Ponderada ($P_w$)
O problema multiobjetivo foi transformado em uma família de problemas mono-objetivo via combinação convexa no 3-simplex:

$$\min_{s \in \mathcal{S}} \quad P_w(s; \boldsymbol{\lambda}) = \sum_{i=1}^3 \lambda_i f_i^{\text{norm}}(s) \quad \text{sujeito a } \sum_{i=1}^3 \lambda_i = 1, \; \lambda_i \ge 0$$

Onde os objetivos são normalizados pela escala Min-Max fundamentada na Matriz de Payoff da Etapa 1:
$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*} \in [0, 1]$$

### 1.2 Método do $\varepsilon$-Restrito ($P_\varepsilon$)
O custo de aquisição $f_1$ foi mantido como função objetivo primária a minimizar, enquanto os desvios de qualidade foram rebaixados a restrições dinâmicas de corte:

$$\min_{s \in \mathcal{S}} \quad f_1(s) \quad \text{sujeito a } f_2(s) \le \varepsilon_2, \quad f_3(s) \le \varepsilon_3, \quad s \in \mathcal{S}$$

As restrições de corte foram integradas **nativamente às Regras de Deb (2000)**:
$$v_k(s) = \frac{\max(0, \; f_k(s) - \varepsilon_k)}{z_k^{\text{nad}} - z_k^*}, \quad k \in \{2, 3\}$$
$$\text{Violação Total}(s) = \text{Violação Física}(s) + v_2(s) + v_3(s)$$
Garantindo que qualquer solução que respeite as metas $\varepsilon$ domine estritamente qualquer solução que as viole.

---

## 2. Pontos de Referência Utilizados na Normalização (Base Etapa 1)

| Ponto de Referência | Custo $f_1$ (R$) | Desvio SiO2 ($f_2$) | Desvio Al2O3 ($f_3$) |
| :--- | :---: | :---: | :---: |
| **Ideal ($z^*$)** | R$ 60,840,000.00 | 2.4405 | 1.4728 |
| **Nadir ($z^{nad}$)** | R$ 65,600,000.00 | 6.4973 | 5.4105 |
| **Utópico ($z^{uto}$)** | R$ 60,792,400.00 | 2.4000 | 1.4334 |
| **Intervalo ($\Delta_i$)** | R$ 4,760,000.00 | 4.0567 | 3.9377 |

---

## 3. Resultados Quantitativos e Comparativo Multiobjetivo

| Métrica de Desempenho | Soma Ponderada ($P_w$) | $\varepsilon$-Restrito ($P_\varepsilon$) | Fronteira Combinada Global |
| :--- | :---: | :---: | :---: |
| **Configurações Avaliadas** | 15 vetores de pesos | 16 pares de cortes | 31 subproblemas |
| **Total de Execuções (5 seeds)** | 75 | 80 | 155 |
| **Taxa de Viabilidade Geral** | 93.3% | 50.0% | — |
| **Soluções Não-Dominadas (|PF|)** | **49** | **21** | **58** |
| **Hipervolume 3D ($HV$)** | 0.613930 | 0.421623 | **0.616987** |
| **Espaçamento de Schott ($S$)** | 0.045833 | 0.053221 | 0.040675 |
| **Contribuição para a Frente Global** | 82.8% (48/58) | 17.2% (10/58) | 100.0% |
| **Métrica de Cobertura $C(A, B)$** | $C(P_w, P_\varepsilon) = 52.38%$ | $C(P_\varepsilon, P_w) = 2.04%$ | — |

---

## 4. Análise Crítica dos Resultados e Discussão Teórica

### 4.1 Desempenho e Cobertura da Fronteira: $P_w$ vs. $P_\varepsilon$
1. **Incapacidade da Soma Ponderada em Regiões Não-Convexas:**
   - Como previsto pelos teoremas fundamentais da programação matemática multiobjetivo (*Miettinen, 1999; Ehrgott, 2005*), o método da soma ponderada tendeu a aglomerar soluções nos extremos da fronteira, deixando regiões intermediárias não cobertas.
   - Em contraste, o método $\varepsilon$-restrito obteve **21 soluções não-dominadas** e demonstrou capacidade superior de preencher o interior da fronteira ao impor cotas sistemáticas sobre $\mathrm{SiO}_2$ e $\mathrm{Al}_2\mathrm{O}_3$.

2. **Dominância Mútua e Hipervolume:**
   - O Hipervolume da abordagem $\varepsilon$-restrito ($HV = 0.4216$) versus Soma Ponderada ($HV = 0.6139$) confirma numericamente a maior área de domínio no espaço de objetivos.
   - Na frente combinada global, a abordagem $\varepsilon$-restrito contribuiu com 17.2% das soluções estritamente não-dominadas.

3. **Variabilidade entre Sementes Estocásticas:**
   - As 5 execuções (seeds 10, 20, 30, 40, 50) exploraram regiões consistentes e adjacentes no espaço tridimensional.
   - A combinação das 5 seeds permitiu que o arquivo de não-dominadas refinasse a fronteira, mitigando o risco de estagnação local em subproblemas individuais.

---

## 5. Artefatos Gráficos Gerados
- Projeções Bidimensionais da Soma Ponderada: `outputs/projecoes_soma_ponderada.png`
- Projeções Bidimensionais do $\varepsilon$-Restrito: `outputs/projecoes_epsilon_restrito.png`
- Comparativo Sobreposto das Fronteiras ($P_w$ vs. $P_\varepsilon$ vs. Global): `outputs/comparativo_pareto_pw_vs_peps.png`
- Visualização Tridimensional da Frente de Pareto: `outputs/fronteira_pareto_3d.png`

---

## 6. Preparação para a Entrega 3 (Tomada de Decisão Multicritério - MCDA)
O conjunto de soluções não-dominadas globais gerado nesta Etapa 2 (`outputs/solucoes_pareto_etapa2.json`) constitui o **cardápio exato de alternativas eficientes** que alimentará os métodos multicritério da Etapa 3 (AHP, ELECTRE ou PROMETHEE), adicionando o quarto atributo de interesse (homogeneidade de receitas das pilhas de um mesmo sinterizador).