# -*- coding: utf-8 -*-
"""
neighborhoods.py
Estruturas de vizinhança usadas pela metaheurística VNS/GVNS.

N1 - Troca simples (single swap):
    Escolhe uma pilha p e uma posição i do vetor de composição, e substitui
    o minério dessa posição por outro minério elegível para o grupo de p
    (diferente do atual, escolhido aleatoriamente).

N2 - Troca entre pilhas (cross-pile exchange):
    Escolhe duas pilhas p1 != p2 e uma posição em cada uma; se os dois
    minérios envolvidos forem mutuamente compatíveis (o minério de p1 é
    elegível para o grupo de p2 e vice-versa), troca os dois minérios de
    posição. Note que quando p1 e p2 pertencem ao mesmo grupo a troca é
    sempre compatível; quando pertencem a grupos diferentes, só é aceita se
    ambos os minérios forem "duplamente elegíveis" (M1-M7, M13-M15).

N3 - Substituição dupla direcionada (targeted double-replacement):
    Vizinhança de perturbação mais "forte" (passo maior), com lógica
    distinta de N1/N2: escolhe uma pilha p e DUAS posições i≠j dessa pilha
    simultaneamente, substituindo os minérios de ambas por dois minérios
    elegíveis sorteados aleatoriamente (iguais ou diferentes entre si e dos
    originais). Ao alterar duas posições de uma só vez, N3 produz um
    deslocamento maior no espaço de soluções do que N1/N2 (que alteram uma
    única posição, ou trocam duas posições mantendo os mesmos minérios em
    uso), sendo útil como nível de agitação (shaking) mais intenso dentro
    do VNS/GVNS.

Todos os operadores retornam uma NOVA solução (cópia), preservando a
original, para permitir avaliação e critério de aceitação.
"""

import copy
import random
from data import PILES, PILE_IDS, ELIGIBLE


def _copy(sol):
    return {p: list(comp) for p, comp in sol.items()}


def n1_troca_simples(sol, rng: random.Random):
    """N1: troca o minério de uma posição por outro elegível diferente."""
    novo = _copy(sol)
    p = rng.choice(PILE_IDS)
    grp = PILES[p]["grupo"]
    i = rng.randrange(len(novo[p]))
    atual = novo[p][i]
    opcoes = [m for m in ELIGIBLE[grp] if m != atual]
    if not opcoes:
        return novo  # não deveria ocorrer (há >=15 minérios elegíveis por grupo)
    novo[p][i] = rng.choice(opcoes)
    return novo


def n2_troca_entre_pilhas(sol, rng: random.Random, max_tentativas=20):
    """N2: troca minérios entre posições de duas pilhas distintas, se compatível."""
    novo = _copy(sol)
    for _ in range(max_tentativas):
        p1, p2 = rng.sample(PILE_IDS, 2)
        i1 = rng.randrange(len(novo[p1]))
        i2 = rng.randrange(len(novo[p2]))
        m1, m2 = novo[p1][i1], novo[p2][i2]
        grp1, grp2 = PILES[p1]["grupo"], PILES[p2]["grupo"]
        compat = (m1 in ELIGIBLE[grp2]) and (m2 in ELIGIBLE[grp1])
        if compat and m1 != m2:
            novo[p1][i1], novo[p2][i2] = m2, m1
            return novo
    return novo  # se não achou par compatível em max_tentativas, devolve inalterado


def n3_substituicao_dupla(sol, rng: random.Random):
    """N3: substitui os minérios de DUAS posições de uma mesma pilha por
    novos minérios sorteados aleatoriamente (passo de perturbação maior)."""
    novo = _copy(sol)
    p = rng.choice(PILE_IDS)
    grp = PILES[p]["grupo"]
    n = len(novo[p])
    if n < 2:
        return n1_troca_simples(sol, rng)
    i, j = rng.sample(range(n), 2)
    novo[p][i] = rng.choice(ELIGIBLE[grp])
    novo[p][j] = rng.choice(ELIGIBLE[grp])
    return novo


NEIGHBORHOODS = [n1_troca_simples, n2_troca_entre_pilhas, n3_substituicao_dupla]
NEIGHBORHOOD_NAMES = ["N1 (troca simples)", "N2 (troca entre pilhas)", "N3 (substituicao dupla)"]
