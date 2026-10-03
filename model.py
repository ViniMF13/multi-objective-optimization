# -*- coding: utf-8 -*-
"""
model.py
Representação da solução, cálculo das funções objetivo f1, f2, f3 e
verificação/penalização das restrições do problema.

REPRESENTAÇÃO DA SOLUÇÃO
-------------------------
Uma solução é um dicionário:
    sol = {"P1": [m_1, m_2, ..., m_{n1}], "P2": [...], ..., "P10": [...]}
onde cada lista tem comprimento igual ao número de caminhões (posições) da
pilha (massa_pilha / 2 kt), e cada elemento é o id do minério (string,
ex. "M7") alocado àquele caminhão/posição daquela pilha. Cada posição i da
pilha p só pode conter um minério elegível para o grupo de p.

Como todos os caminhões têm a mesma massa (2 kt), a restrição de
atendimento EXATO da massa de cada pilha é satisfeita por construção
(o vetor sempre tem o comprimento correto): não é necessária uma variável
de folga. Da mesma forma, a elegibilidade (minério só alimenta os
equipamentos estipulados) é garantida por construção, pois o domínio de
cada posição já é restrito à lista de minérios elegíveis do grupo da pilha.

As duas restrições que realmente podem ser violadas pelos operadores de
busca (vizinhanças) são:
  (i)  limites de disponibilidade total de cada minério (mínima e máxima);
  (ii) limites min/max de SiO2 e Al2O3 de cada pilha (o FeT nunca é ativo
       neste case, pois os limites são 0-100%).
Essas violações são tratadas por PENALIZAÇÃO (ver função `avaliar`).
"""

from data import MINERALS, PILES, PILE_IDS, GROUP_BOUNDS, ELIGIBLE, TRUCK_MASS_T

# Peso da penalidade (grande o suficiente para dominar qualquer diferença
# nas funções objetivo, que estão em escalas bem menores).
PENALTY_WEIGHT = 1.0e7


def pile_group(pid):
    return PILES[pid]["grupo"]


def pile_quality(comp):
    """
    Calcula a qualidade média (FeT, SiO2, Al2O3) de uma pilha a partir da
    sua lista de minérios `comp` (um por caminhão/posição). Como todos os
    caminhões têm a mesma massa, a média ponderada se reduz à média
    aritmética simples sobre as posições.
    """
    n = len(comp)
    fet = sum(MINERALS[m]["FeT"] for m in comp) / n
    sio2 = sum(MINERALS[m]["SiO2"] for m in comp) / n
    al2o3 = sum(MINERALS[m]["Al2O3"] for m in comp) / n
    return fet, sio2, al2o3


def pile_cost(comp):
    """Custo de aquisição dos minérios de uma pilha (R$)."""
    return sum(MINERALS[m]["preco"] for m in comp) * TRUCK_MASS_T


# ------------------------- Funções objetivo -----------------------------

def f1_custo(sol):
    """f1: custo total de aquisição dos minérios (R$), a MINIMIZAR."""
    return sum(pile_cost(sol[p]) for p in PILE_IDS)


def f2_desvio_sio2(sol):
    """f2: soma dos desvios quadráticos de SiO2 de cada pilha em relação
    ao alvo do grupo, a MINIMIZAR."""
    total = 0.0
    for p in PILE_IDS:
        grp = pile_group(p)
        _, sio2, _ = pile_quality(sol[p])
        alvo = GROUP_BOUNDS[grp]["SiO2_alvo"]
        total += (sio2 - alvo) ** 2
    return total


def f3_desvio_al2o3(sol):
    """f3: soma dos desvios quadráticos de Al2O3 de cada pilha em relação
    ao alvo do grupo, a MINIMIZAR."""
    total = 0.0
    for p in PILE_IDS:
        grp = pile_group(p)
        _, _, al2o3 = pile_quality(sol[p])
        alvo = GROUP_BOUNDS[grp]["Al2O3_alvo"]
        total += (al2o3 - alvo) ** 2
    return total


OBJECTIVES = {"f1": f1_custo, "f2": f2_desvio_sio2, "f3": f3_desvio_al2o3}


# --------------------------- Restrições ----------------------------------

def violacao_disponibilidade(sol):
    """
    Retorna a soma das violações (em número de caminhões) dos limites de
    disponibilidade mínima e máxima de cada minério, somadas sobre as duas
    pilhas/grupos (a disponibilidade é definida por minério, não por pilha).
    """
    uso = {m: 0 for m in MINERALS}
    for p in PILE_IDS:
        for m in sol[p]:
            uso[m] += 1
    viol = 0.0
    for m, dados in MINERALS.items():
        if uso[m] < dados["disp_min"]:
            viol += (dados["disp_min"] - uso[m])
        if uso[m] > dados["disp_max"]:
            viol += (uso[m] - dados["disp_max"])
    return viol


def violacao_qualidade(sol):
    """
    Retorna a soma das violações dos limites min/max de SiO2 e Al2O3 de
    cada pilha (em pontos percentuais). O FeT nunca é violado neste case
    (limites 0-100%) e por isso não é verificado.
    """
    viol = 0.0
    for p in PILE_IDS:
        grp = pile_group(p)
        b = GROUP_BOUNDS[grp]
        _, sio2, al2o3 = pile_quality(sol[p])
        if sio2 < b["SiO2_min"]:
            viol += (b["SiO2_min"] - sio2)
        if sio2 > b["SiO2_max"]:
            viol += (sio2 - b["SiO2_max"])
        if al2o3 < b["Al2O3_min"]:
            viol += (b["Al2O3_min"] - al2o3)
        if al2o3 > b["Al2O3_max"]:
            viol += (al2o3 - b["Al2O3_max"])
    return viol


def penalidade_total(sol):
    return PENALTY_WEIGHT * (violacao_disponibilidade(sol) + violacao_qualidade(sol))


def is_feasible(sol, tol=1e-9):
    return (violacao_disponibilidade(sol) <= tol) and (violacao_qualidade(sol) <= tol)


def avaliar(sol, objetivo):
    """
    Fitness usado pela metaheurística: f_objetivo(sol) + penalidade das
    restrições violadas (disponibilidade e qualidade min/max).
    `objetivo` in {"f1","f2","f3"}.
    Política de inviabilidade adotada: ACEITAÇÃO TEMPORÁRIA da
    infactibilidade mediante penalização.
    """
    f = OBJECTIVES[objetivo](sol)
    pen = penalidade_total(sol)
    return f + pen


def resumo_solucao(sol):
    """Retorna um dicionário com as métricas relevantes de uma solução."""
    return dict(
        f1=f1_custo(sol),
        f2=f2_desvio_sio2(sol),
        f3=f3_desvio_al2o3(sol),
        viol_disp=violacao_disponibilidade(sol),
        viol_qual=violacao_qualidade(sol),
        factivel=is_feasible(sol),
    )
