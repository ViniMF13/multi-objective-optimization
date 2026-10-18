# -*- coding: utf-8 -*-
"""
neighborhoods.py
Estruturas de vizinhança canônicas para o GVNS com o Princípio de Dominação de Deb:
1. Operadores de Perturbação (Shaking) com intensidades progressivas k = 1, 2, 3:
   - Shaking N1: Substituição Simples em 1 pilha (d_H = 1)
   - Shaking N2: Troca Cruzada entre 2 pilhas distintas (d_H = 2, conservativa de estoque)
   - Shaking N3: Substituição Tripla Distribuída em 3 pilhas distintas (d_H = 3, alcance sistêmico)
2. Operadores de Descida de Busca Local (VND) guiados pelo Princípio de Deb:
   - Descida N1 (Substituição Simples)
   - Descida N2 (Troca Entre Pilhas)
   - Descida N3 (Substituição Tripla Distribuída)
"""

import copy
import random
from data import PILES, PILE_IDS, ELIGIBLE, MINERALS
from model import avaliar, resumo_solucao, eh_melhor_deb, violacao_total


def copy_solution(sol):
    """Retorna uma cópia profunda (deep copy) da estrutura de dados da solução."""
    return {p: list(comp) for p, comp in sol.items()}


# ---------------------------------------------------------------------------
# 1. Operadores de Perturbação (Shaking)
# ---------------------------------------------------------------------------

def shaking_n1(sol, rng: random.Random = None):
    """
    Shaking N1 (Perturbação Leve, Distância de Hamming = 1):
    Sorteia 1 pilha p e 1 posição i, substituindo o minério atual
    por outro minério elegível diferente sorteado aleatoriamente.
    """
    if rng is None:
        rng = random.Random()
    novo = copy_solution(sol)
    p = rng.choice(PILE_IDS)
    grp = PILES[p]["grupo"]
    i = rng.randrange(len(novo[p]))
    atual = novo[p][i]
    opcoes = [m for m in ELIGIBLE[grp] if m != atual]
    if opcoes:
        novo[p][i] = rng.choice(opcoes)
    return novo


def shaking_n2(sol, rng: random.Random = None, max_tentativas=50):
    """
    Shaking N2 (Perturbação Moderada, Distância de Hamming = 2):
    Sorteia 2 pilhas distintas p1 != p2 e troca minérios de posições i1, i2
    se forem mutuamente compatíveis (preserva o estoque global de minérios).
    """
    if rng is None:
        rng = random.Random()
    novo = copy_solution(sol)
    for _ in range(max_tentativas):
        p1, p2 = rng.sample(PILE_IDS, 2)
        i1 = rng.randrange(len(novo[p1]))
        i2 = rng.randrange(len(novo[p2]))
        m1, m2 = novo[p1][i1], novo[p2][i2]
        grp1, grp2 = PILES[p1]["grupo"], PILES[p2]["grupo"]
        if (m1 in ELIGIBLE[grp2]) and (m2 in ELIGIBLE[grp1]) and (m1 != m2):
            novo[p1][i1], novo[p2][i2] = m2, m1
            return novo
    # Se não encontrar par compatível após max_tentativas, aplica N1 como fallback
    return shaking_n1(sol, rng)


def shaking_n3(sol, rng: random.Random = None):
    """
    Shaking N3 (Perturbação Forte e Distribuída, Distância de Hamming = 3):
    Sorteia 3 pilhas distintas p1, p2, p3 e, em cada uma delas, substitui 1 caminhão
    por outro minério elegível diferente do atual.
    Afeta 30% das pilhas do sistema de forma balanceada, com 100% de elegibilidade por construção.
    """
    if rng is None:
        rng = random.Random()
    novo = copy_solution(sol)
    pilhas_sorteadas = rng.sample(PILE_IDS, 3)
    for p in pilhas_sorteadas:
        grp = PILES[p]["grupo"]
        i = rng.randrange(len(novo[p]))
        atual = novo[p][i]
        opcoes = [m for m in ELIGIBLE[grp] if m != atual]
        if opcoes:
            novo[p][i] = rng.choice(opcoes)
    return novo


def shaking_k(sol, k: int, rng: random.Random = None):
    """
    Aplica perturbação de intensidade k (1-indexado):
    k = 1: N1 (Substituição Simples - 1 pilha)
    k = 2: N2 (Troca Entre 2 Pilhas - 2 pilhas)
    k = 3: N3 (Substituição Tripla Distribuída - 3 pilhas)
    k > 3: k passos acumulados de perturbação
    """
    if rng is None:
        rng = random.Random()
    if k == 1:
        return shaking_n1(sol, rng)
    elif k == 2:
        return shaking_n2(sol, rng)
    elif k == 3:
        return shaking_n3(sol, rng)
    else:
        atual = sol
        for _ in range(k):
            atual = shaking_n1(atual, rng)
        return atual


# ---------------------------------------------------------------------------
# 2. Operadores de Busca Local (Descida First-Improvement com Regras de Deb)
# ---------------------------------------------------------------------------

def descida_n1(sol, objetivo, rng: random.Random = None, max_evals=100, max_sem_melhora=30):
    """
    Busca local em N1 (Substituição Simples em 1 Pilha):
    Sorteia posições e testa minérios elegíveis substitutos via First-Improvement estocástico.
    Critério de Aceitação: Princípio de Dominação de Restrições de Deb (eh_melhor_deb).

    Retorna: (solucao_final, valor_final, avaliacoes_realizadas, houve_melhora, historico_transicoes)
    """
    if rng is None:
        rng = random.Random()
    atual = copy_solution(sol)
    val_atual = avaliar(atual, objetivo)
    avals = 0
    melhorou_global = False
    transicoes = []
    sem_melhora = 0
    limite = min(max_sem_melhora, max_evals)

    while sem_melhora < limite and avals < max_evals:
        p = rng.choice(PILE_IDS)
        grp = PILES[p]["grupo"]
        i = rng.randrange(len(atual[p]))
        m_antigo = atual[p][i]

        opcoes = [m for m in ELIGIBLE[grp] if m != m_antigo]
        if not opcoes:
            sem_melhora += 1
            continue

        m_candidato = rng.choice(opcoes)
        candidato = copy_solution(atual)
        candidato[p][i] = m_candidato
        avals += 1

        if eh_melhor_deb(candidato, atual, objetivo):
            v = avaliar(candidato, objetivo)
            delta = v - val_atual
            transicoes.append({
                "tipo": "N1 (Substituição Simples)",
                "pilha": p,
                "posicao": i,
                "de": m_antigo,
                "para": m_candidato,
                "val_antes": val_atual,
                "val_depois": v,
                "delta": delta,
            })
            atual = candidato
            val_atual = v
            melhorou_global = True
            sem_melhora = 0
        else:
            sem_melhora += 1

    return atual, val_atual, avals, melhorou_global, transicoes


def descida_n2(sol, objetivo, rng: random.Random = None, max_evals=100, max_sem_melhora=25):
    """
    Busca local em N2 (Troca Entre 2 Pilhas Distintas):
    Explora trocas compatíveis entre pilhas distintas p1 != p2.
    Invariante: 100% conservativo de disponibilidade global.
    Critério de Aceitação: Princípio de Dominação de Restrições de Deb (eh_melhor_deb).

    Retorna: (solucao_final, valor_final, avaliacoes_realizadas, houve_melhora, historico_transicoes)
    """
    if rng is None:
        rng = random.Random()
    atual = copy_solution(sol)
    val_atual = avaliar(atual, objetivo)
    avals = 0
    melhorou_global = False
    transicoes = []
    sem_melhora = 0
    limite = min(max_sem_melhora, max_evals)

    while sem_melhora < limite and avals < max_evals:
        p1, p2 = rng.sample(PILE_IDS, 2)
        i1 = rng.randrange(len(atual[p1]))
        i2 = rng.randrange(len(atual[p2]))
        m1, m2 = atual[p1][i1], atual[p2][i2]
        grp1, grp2 = PILES[p1]["grupo"], PILES[p2]["grupo"]

        if (m1 in ELIGIBLE[grp2]) and (m2 in ELIGIBLE[grp1]) and (m1 != m2):
            candidato = copy_solution(atual)
            candidato[p1][i1], candidato[p2][i2] = m2, m1
            avals += 1

            if eh_melhor_deb(candidato, atual, objetivo):
                v = avaliar(candidato, objetivo)
                delta = v - val_atual
                transicoes.append({
                    "tipo": "N2 (Troca Entre Pilhas)",
                    "p1": p1, "i1": i1, "m1": m1,
                    "p2": p2, "i2": i2, "m2": m2,
                    "val_antes": val_atual,
                    "val_depois": v,
                    "delta": delta,
                })
                atual = candidato
                val_atual = v
                melhorou_global = True
                sem_melhora = 0
            else:
                sem_melhora += 1
        else:
            sem_melhora += 1

    return atual, val_atual, avals, melhorou_global, transicoes


def descida_n3(sol, objetivo, rng: random.Random = None, max_evals=80, max_sem_melhora=20):
    """
    Busca local em N3 (Substituição Tripla Distribuída em 3 Pilhas Distintas):
    Sorteia trincas de pilhas distintas p1, p2, p3 e tenta substituir simultaneamente
    1 caminhão em cada uma delas por novos minérios elegíveis.
    Critério de Aceitação: Princípio de Dominação de Restrições de Deb (eh_melhor_deb).

    Retorna: (solucao_final, valor_final, avaliacoes_realizadas, houve_melhora, historico_transicoes)
    """
    if rng is None:
        rng = random.Random()
    atual = copy_solution(sol)
    val_atual = avaliar(atual, objetivo)
    avals = 0
    melhorou_global = False
    transicoes = []
    sem_melhora = 0
    limite = min(max_sem_melhora, max_evals)

    while sem_melhora < limite and avals < max_evals:
        pilhas = rng.sample(PILE_IDS, 3)
        candidato = copy_solution(atual)
        mudancas = []

        for p in pilhas:
            grp = PILES[p]["grupo"]
            i = rng.randrange(len(candidato[p]))
            m_ant = candidato[p][i]
            opcoes = [m for m in ELIGIBLE[grp] if m != m_ant]
            if opcoes:
                m_novo = rng.choice(opcoes)
                candidato[p][i] = m_novo
                mudancas.append((p, i, m_ant, m_novo))

        if len(mudancas) == 3:
            avals += 1
            if eh_melhor_deb(candidato, atual, objetivo):
                v = avaliar(candidato, objetivo)
                delta = v - val_atual
                transicoes.append({
                    "tipo": "N3 (Tripla Distribuída)",
                    "pilhas": [m[0] for m in mudancas],
                    "mudancas": mudancas,
                    "val_antes": val_atual,
                    "val_depois": v,
                    "delta": delta,
                })
                atual = candidato
                val_atual = v
                melhorou_global = True
                sem_melhora = 0
            else:
                sem_melhora += 1
        else:
            sem_melhora += 1

    return atual, val_atual, avals, melhorou_global, transicoes


# Aliases e compatibilidade
_copy = copy_solution
n1_troca_simples = shaking_n1
n2_troca_entre_pilhas = shaking_n2
n3_substituicao_dupla = shaking_n3

SHAKING_OPERATORS = [shaking_n1, shaking_n2, shaking_n3]
LOCAL_SEARCH_OPERATORS = [descida_n1, descida_n2, descida_n3]
NEIGHBORHOOD_NAMES = [
    "N1 (Substituição simples)",
    "N2 (Troca entre 2 pilhas)",
    "N3 (Substituição tripla distribuída)",
]
NEIGHBORHOODS = SHAKING_OPERATORS
