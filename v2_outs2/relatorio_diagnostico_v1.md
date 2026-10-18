# Relatório Diagnóstico de Otimização Mono-Objetivo (Modelo Atual v1)

Este relatório consolida a avaliação da Entrega 1 com análise crítica de viabilidade, escala da função de penalidade e proximidade aos alvos por pilha.

## 1. Estatísticas das Execuções (Separação Viáveis vs Geral)

| Objetivo | Runs | Viáveis | Taxa (%) | Min Viável | Média Viável | Std Viável | Min Geral (Puro) | Média Geral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **f1** (R$) | 5 | 3 | 60% | 61,440,000.00 | 61,626,666.67 | 132,999.58 | 60,880,000.00 | 61,404,000.00 |
| **f2** ((% SiO2)²) | 5 | 5 | 100% | 2.65 | 2.89 | 0.23 | 2.65 | 2.89 |
| **f3** ((% Al2O3)²) | 5 | 4 | 80% | 1.77 | 1.84 | 0.07 | 1.77 | 1.88 |

> **Nota Crítica sobre Viabilidade:**
> Em **f1**, quando a busca aceita pequenas infrações de qualidade (ex.: teores 0,01% fora do limite), > a redução financeira (até R$ 380 mil por caminhão) compensa a penalidade (R$ 100 mil com W=10⁷). > Assim, o 'mínimo geral' pode ser uma solução inviável ilegítima. > A métrica correta de benchmark deve ser o **Min Viável**.

## 2. Detalhamento das Execuções Inviáveis

| Objetivo | Seed | Valor Puro | Penalidade Paga | Violação Disp. (caminhões) | Violação Qualidade (%) | Falhas Identificadas |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| f1 | seed=10 | 61,260,000.00 | 52,631.58 | 0.0 | 0.0053% | P10 SiO2=6.505 (max: 6.5) |
| f1 | seed=40 | 60,880,000.00 | 143,540.67 | 0.0 | 0.0144% | P2 SiO2=6.209 (max: 6.2); P10 SiO2=6.505 (max: 6.5) |
| f3 | seed=10 | 2.07 | 400,000.00 | 0.0 | 0.0400% | P6 Al2O3=2.240 (max: 2.2) |

## 3. Melhores Soluções Selecionadas (Critério de Factibilidade)

| Objetivo | Seed | Status | Valor Puro | Penalidade | f1 (R$) | f2 (SiO2) | f3 (Al2O3) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **f1** | seed=20 | Viável (Ótimo factível encontrado) | 61,440,000.0000 | 0.00 | R$ 61,440,000 | 5.2273 | 5.1206 |
| **f2** | seed=10 | Viável (Ótimo factível encontrado) | 2.6522 | 0.00 | R$ 64,420,000 | 2.6522 | 3.0095 |
| **f3** | seed=40 | Viável (Ótimo factível encontrado) | 1.7687 | 0.00 | R$ 65,580,000 | 4.9387 | 1.7687 |

## 4. Avaliação da Proximidade aos Alvos por Pilha

**Critério Adotado:**
- **Excelente:** Desvio absoluto $|teor - alvo| \le 0,10\%$
- **Bom:** $0,10\% < |desvio| \le 0,25\%$
- **Afastado:** $|desvio| > 0,25\%$

### Proximidade para f2 (SiO2) na Melhor Solução (seed=10)
- **Valor Agregado (f2):** 2.6522
- **Desvio Médio por Pilha:** 0.5074%
- **Pior Pilha (Desvio Máximo):** 0.6526%
- **Uniformidade (Desvio Padrão entre pilhas):** 0.0880%
- **Distribuição das Pilhas:** 0 Excelente(s), 0 Boa(s), 10 Afastada(s), 0 Inviável(is)

### Proximidade para f3 (Al2O3) na Melhor Solução (seed=40)
- **Valor Agregado (f3):** 1.7687
- **Desvio Médio por Pilha:** 0.4017%
- **Pior Pilha (Desvio Máximo):** 0.6115%
- **Uniformidade (Desvio Padrão entre pilhas):** 0.1247%
- **Distribuição das Pilhas:** 0 Excelente(s), 1 Boa(s), 9 Afastada(s), 0 Inviável(is)
