# -*- coding: utf-8 -*-
"""
vns.py
Implementação canônica das metaheurísticas:
1. VND (Variable Neighborhood Descent):
   Procedimento de refinamento (ou busca local) multi-vizinhança determinística/sistemática que explora em cascata
   as vizinhanças de descida N1 (substituição simples), N2 (troca entre pilhas) e N3 (substituição dupla).
   Ao encontrar melhora em N_l, reinicia em N1 (l = 1); só avança para N_{l+1} quando estagna.

2. GVNS (General Variable Neighborhood Search):
   Metaheurística completa integrando:
   - Heurística construtiva inicial
   - VND inicial para atingir o primeiro ótimo local
   - Shaking estocástico com vizinhanças de perturbação N_k (k = 1...k_max)
   - Busca local via VND sobre cada solução perturbada
   - Critério de aceitação de melhora estrita (move-or-not) com reinício de k = 1 em caso de sucesso
"""

import random
from model import avaliar, resumo_solucao, eh_melhor_deb
from construction import construir_solucao
from neighborhoods import (
    copy_solution,
    SHAKING_OPERATORS,
    LOCAL_SEARCH_OPERATORS,
    descida_n1,
    descida_n2,
    descida_n3,
)


def vnd(sol, objetivo, rng=None, max_evals=1000, verbose=False, trace=None):
    """
    Variable Neighborhood Descent (VND):
    Busca local determinística que explora um conjunto ordenado de vizinhanças
    de descida L = [N1, N2, N3].

    Algoritmo:
    l <- 0
    enquanto l < l_max e avaliacoes < max_evals:
        s' <- LocalSearch(s, N_l)
        se avaliar(s') < avaliar(s):
            s <- s'
            l <- 0    # Sucesso: reinicia na vizinhança mais fina (N1)
        senão:
            l <- l + 1 # Falha: avança para a próxima vizinhança

    Retorna: (solucao_final, valor_final, total_avaliacoes, estatisticas_vnd)
    """
    if rng is None:
        rng = random.Random(0)

    s = copy_solution(sol)
    val_s = avaliar(s, objetivo)
    total_avals = 0

    stats = {
        "transicoes": [],
        "avals_por_vizinhanca": [0] * len(LOCAL_SEARCH_OPERATORS),
        "melhoras_por_vizinhanca": [0] * len(LOCAL_SEARCH_OPERATORS),
        "sucesso_global": False,
        "valor_inicial": val_s,
        "valor_final": val_s,
    }

    l = 0
    l_max = len(LOCAL_SEARCH_OPERATORS)

    while l < l_max and total_avals < max_evals:
        orcamento_restante = max_evals - total_avals
        operador_ls = LOCAL_SEARCH_OPERATORS[l]
        nome_viz = f"N{l+1}"

        novo_s, novo_val, avals_ls, melhorou, log_trans = operador_ls(
            s, objetivo, rng, max_evals=orcamento_restante
        )
        total_avals += avals_ls
        stats["avals_por_vizinhanca"][l] += avals_ls

        if melhorou and eh_melhor_deb(novo_s, s, objetivo):
            delta = novo_val - val_s
            stats["melhoras_por_vizinhanca"][l] += 1
            stats["sucesso_global"] = True

            transicao_info = {
                "vizinhanca": nome_viz,
                "l_index": l,
                "val_antes": val_s,
                "val_depois": novo_val,
                "delta": delta,
                "avals_gastas": avals_ls,
                "detalhes": log_trans,
            }
            stats["transicoes"].append(transicao_info)

            if verbose:
                print(f"[VND] Melhora em {nome_viz}: {val_s:.4f} -> {novo_val:.4f} (delta={delta:.4f})")

            if trace is not None and hasattr(trace, "append"):
                trace.append(f"[VND] Melhora em {nome_viz}: {val_s:.4f} -> {novo_val:.4f}")

            s = novo_s
            val_s = novo_val
            l = 0  # REINICIA na primeira estrutura de vizinhança (N1)!
        else:
            # Não houve melhora nesta vizinhança: avança para a próxima
            l += 1

    stats["valor_final"] = val_s
    return s, val_s, total_avals, stats


def busca_local_n1(sol, objetivo, rng, max_iter_sem_melhora=25):
    """
    Função de compatibilidade retroativa para código legado e testes existentes.
    Executa descida em N1.
    """
    s_desc, val_desc, avals, melhorou, _ = descida_n1(
        sol, objetivo, rng, max_evals=max_iter_sem_melhora * 10
    )
    return s_desc, val_desc, avals


def gvns(objetivo, seed=0, max_evals=5000, k_max=3, ls_evals_limit=200, verbose=False, trace=None):
    """
    General Variable Neighborhood Search (GVNS) Canônico:
    1. s <- Heurística Construtiva (solução inicial)
    2. s <- VND(s)  # Primeiro ótimo local
    3. Enquanto avals < max_evals:
           k <- 0
           Enquanto k < k_max e avals < max_evals:
               s' <- Shaking(s, k)       # Perturbação estocástica
               s'' <- VND(s', ...)       # Busca Local via VND com N1, N2, N3
               se avaliar(s'') < avaliar(s):
                   s <- s''; k <- 0      # Sucesso: aceita e volta para k=0
               senão:
                   k <- k + 1            # Falha: intensifica perturbação
               se avaliar(s) < avaliar(melhor):
                   melhor <- s

    Retorna: melhor_solucao, melhor_valor, historico_convergencia
    onde `historico` é uma lista de tuplas (num_avaliacoes, melhor_valor).
    """
    rng = random.Random(seed)

    # 1. Solução inicial pela heurística construtiva
    s0 = construir_solucao(seed=seed)
    val_s0 = avaliar(s0, objetivo)

    avals = 1
    historico = [(avals, val_s0)]

    # 2. Descida inicial com VND sobre a solução construtiva para encontrar o primeiro ótimo local
    s, val_s, avals_init_vnd, _ = vnd(
        s0, objetivo, rng=rng, max_evals=min(ls_evals_limit, max_evals - avals), verbose=verbose
    )
    avals += avals_init_vnd

    melhor = s
    val_melhor = val_s
    historico.append((avals, val_melhor))

    if verbose:
        print(f"[{objetivo}] GVNS Inicial: Const={val_s0:.4f} -> VND_Inicial={val_s:.4f} (avals={avals})")

    # 3. Loop Principal do GVNS
    while avals < max_evals:
        k = 0
        while k < k_max and avals < max_evals:
            # (a) Shaking / Perturbação Estocástica no nível k
            operador_shake = SHAKING_OPERATORS[k % len(SHAKING_OPERATORS)]
            s_linha = operador_shake(s, rng)
            avals += 1

            # (b) Busca Local sistemática com VND a partir de s_linha
            orcamento_vnd = min(ls_evals_limit, max_evals - avals)
            s_2linhas, val_2linhas, avals_vnd, stats_vnd = vnd(
                s_linha, objetivo, rng=rng, max_evals=orcamento_vnd, verbose=False
            )
            avals += avals_vnd

            # (c) Critério de Aceitação (Move or Not) baseado em Deb
            if eh_melhor_deb(s_2linhas, s, objetivo):
                s = s_2linhas
                val_s = val_2linhas
                k = 0  # Sucesso: reinicia na perturbação mínima
                if verbose:
                    print(f"[{objetivo}] GVNS Sucesso: Novo incumbent = {val_s:.4f} (avals={avals})")
            else:
                k += 1  # Falha: intensifica o shaking para o próximo nível

            # Atualiza o melhor global
            if eh_melhor_deb(s, melhor, objetivo):
                melhor = s
                val_melhor = val_s

            historico.append((avals, val_melhor))

    return melhor, val_melhor, historico


if __name__ == "__main__":
    for obj in ["f1", "f2", "f3"]:
        sol, val, hist = gvns(obj, seed=1, max_evals=1500, verbose=True)
        print(f"\nFinal {obj}: valor={val:.4f} | resumo={resumo_solucao(sol)}")
