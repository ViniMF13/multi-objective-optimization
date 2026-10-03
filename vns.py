# -*- coding: utf-8 -*-
"""
vns.py
Implementação da metaheurística GVNS (General Variable Neighborhood Search)
para as versões mono-objetivo do problema.

PSEUDOCÓDIGO (resumo - ver relatório para a versão completa e comentada)
--------------------------------------------------------------------------
1. s <- heurística construtiva (construction.construir_solucao)
2. melhor <- s
3. enquanto critério de parada não satisfeito:
     k <- 1
     enquanto k <= k_max:
         s' <- Shaking(s, N_k)                 # perturbação aleatória em N_k
         s'' <- BuscaLocal(s', N1)              # descida usando N1 (first-improvement)
         se avaliar(s'') < avaliar(s):
             s <- s''; k <- 1                   # sucesso: reinicia em N1
         senão:
             k <- k + 1                         # falha: intensifica a perturbação
         se avaliar(s) < avaliar(melhor): melhor <- s
         registra (avaliacoes, avaliar(melhor)) na curva de convergência
4. retorna melhor, curva de convergência

Critério de parada: número máximo de avaliações de função (`max_evals`) OU
número máximo de iterações sem melhora no incumbente (`max_sem_melhora`).

Tratamento de inviabilidade: penalização (ver model.avaliar) - soluções
inviáveis são aceitas temporariamente no processo de busca, mas o valor
penalizado sempre domina o valor "puro" do objetivo, direcionando a busca
para a região viável.
"""

import random
from model import avaliar, resumo_solucao
from construction import construir_solucao
from neighborhoods import NEIGHBORHOODS


def busca_local_n1(sol, objetivo, rng, max_iter_sem_melhora=25):
    """
    Busca local (descida) usando a vizinhança N1 (troca simples) com
    estratégia first-improvement: tenta um vizinho aleatório em N1 por vez;
    aceita se melhora; para após `max_iter_sem_melhora` tentativas
    consecutivas sem sucesso. Retorna (solucao, valor, avaliacoes_realizadas).
    """
    from neighborhoods import n1_troca_simples
    atual = sol
    valor_atual = avaliar(atual, objetivo)
    sem_melhora = 0
    avals = 0
    while sem_melhora < max_iter_sem_melhora:
        vizinho = n1_troca_simples(atual, rng)
        v = avaliar(vizinho, objetivo)
        avals += 1
        if v < valor_atual - 1e-9:
            atual, valor_atual = vizinho, v
            sem_melhora = 0
        else:
            sem_melhora += 1
    return atual, valor_atual, avals


def gvns(objetivo, seed=0, max_evals=4000, k_max=3, ls_sem_melhora=20, verbose=False):
    """
    Executa o GVNS para o objetivo informado ("f1", "f2" ou "f3").
    Retorna: melhor_solucao, melhor_valor, historico
    onde `historico` é uma lista de tuplas (num_avaliacoes, melhor_valor_ate_entao),
    usada para plotar a curva de convergência.
    """
    rng = random.Random(seed)

    s = construir_solucao(seed=seed)
    valor_s = avaliar(s, objetivo)
    melhor, valor_melhor = s, valor_s

    avals = 1
    historico = [(avals, valor_melhor)]

    while avals < max_evals:
        k = 0
        while k < k_max and avals < max_evals:
            # ---- Shaking: perturba a solução atual usando a vizinhança N_k ----
            s_linha = NEIGHBORHOODS[k](s, rng)
            avals += 1

            # ---- Busca local a partir de s' usando N1 ----
            s_2linhas, valor_2linhas, avals_ls = busca_local_n1(
                s_linha, objetivo, rng, max_iter_sem_melhora=ls_sem_melhora
            )
            avals += avals_ls

            # ---- Critério de aceitação: melhora estrita (move-or-not) ----
            if valor_2linhas < valor_s - 1e-9:
                s, valor_s = s_2linhas, valor_2linhas
                k = 0  # sucesso: volta para a vizinhança mais "fina"
            else:
                k += 1  # falha: intensifica a perturbação

            if valor_s < valor_melhor - 1e-9:
                melhor, valor_melhor = s, valor_s

            historico.append((avals, valor_melhor))

            if verbose and avals % 200 == 0:
                print(f"[{objetivo}] avals={avals} k={k} melhor={valor_melhor:.4f}")

    return melhor, valor_melhor, historico


if __name__ == "__main__":
    for obj in ["f1", "f2", "f3"]:
        sol, val, hist = gvns(obj, seed=1, max_evals=2000, verbose=False)
        print(obj, "valor final (penalizado) =", val, "| resumo:", resumo_solucao(sol))
