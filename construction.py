# -*- coding: utf-8 -*-
"""
construction.py
Heurística construtiva para gerar a solução inicial.

Estratégia adaptada:
Para cada pilha p (em ordem crescente de índice):
  1. Preenche posições do vetor de composição priorizando os minérios
     elegíveis para o grupo de p que ainda possuem disponibilidade MÍNIMA
     de uso pendente (para tentar satisfazer disp_min o quanto antes,
     evitando penalização por disponibilidade mínima não atingida);
  2. Completa as posições restantes escolhendo, em ordem crescente de
     preço, os minérios elegíveis daquele grupo (sempre o mais barato
     disponível), controlando o uso acumulado de cada minério para não
     ultrapassar sua disponibilidade máxima quando possível.

Observação (conforme o enunciado): esta heurística NÃO garante que os
limites de qualidade (SiO2, Al2O3) sejam respeitados. É esperado que a
solução inicial seja inviável quanto à qualidade; a busca local/VNS trata
essa inviabilidade por penalização (ver model.avaliar).
"""

import random
from data import MINERALS, PILES, PILE_IDS, ELIGIBLE


def construir_solucao(seed=None):
    rng = random.Random(seed)
    uso_atual = {m: 0 for m in MINERALS}

    sol = {}
    # Ordem de preenchimento das pilhas (conforme enunciado, sequencialmente)
    for p in PILE_IDS:
        grp = PILES[p]["grupo"]
        n_trucks = PILES[p]["n_trucks"]
        elegiveis = ELIGIBLE[grp]

        # minérios elegíveis ordenados por preço crescente
        elegiveis_por_preco = sorted(elegiveis, key=lambda m: MINERALS[m]["preco"])

        comp = []

        # 1) prioriza minérios com disponibilidade mínima ainda pendente
        pendentes = [m for m in elegiveis if uso_atual[m] < MINERALS[m]["disp_min"]]
        # também em ordem de preço crescente, para não gastar demais
        pendentes = sorted(pendentes, key=lambda m: MINERALS[m]["preco"])
        for m in pendentes:
            faltam = MINERALS[m]["disp_min"] - uso_atual[m]
            for _ in range(faltam):
                if len(comp) >= n_trucks:
                    break
                comp.append(m)
                uso_atual[m] += 1
            if len(comp) >= n_trucks:
                break

        # 2) completa com os minérios mais baratos elegíveis, respeitando
        #    (quando possível) a disponibilidade máxima já utilizada
        idx = 0
        while len(comp) < n_trucks:
            # tenta encontrar o minério mais barato que ainda não estourou
            candidato = None
            for m in elegiveis_por_preco:
                if uso_atual[m] < MINERALS[m]["disp_max"]:
                    candidato = m
                    break
            if candidato is None:
                # todos estourariam o máximo: usa mesmo assim o mais barato
                # (a violação será penalizada na avaliação da solução)
                candidato = elegiveis_por_preco[idx % len(elegiveis_por_preco)]
                idx += 1
            comp.append(candidato)
            uso_atual[candidato] += 1

        sol[p] = comp

    return sol


if __name__ == "__main__":
    from model import resumo_solucao
    sol = construir_solucao(seed=42)
    for p, comp in sol.items():
        print(p, comp)
    print(resumo_solucao(sol))
