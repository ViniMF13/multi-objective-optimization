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


def violacao_total(sol):
    """Retorna a soma das violações de disponibilidade (caminhões) e qualidade (pontos %)."""
    return violacao_disponibilidade(sol) + violacao_qualidade(sol)


def penalidade_total(sol):
    return PENALTY_WEIGHT * violacao_total(sol)


def is_feasible(sol, tol=1e-9):
    return (violacao_disponibilidade(sol) <= tol) and (violacao_qualidade(sol) <= tol)


def fitness_deb(sol, objetivo):
    """
    Retorna a chave lexicográfica canônica de Deb (2000) para ordenação e seleção:
    - Se a solução for viável:   (0, f_objetivo(sol), 0.0)
    - Se a solução for inviável: (1, violacao_total(sol), f_objetivo(sol))

    Propriedades matemáticas do Princípio de Dominação de Restrições:
    1. Factibilidade tem prioridade absoluta: qualquer solução viável (0) domina
       qualquer solução inviável (1), independentemente do custo aparente.
    2. Entre viáveis: menor valor da função objetivo vence.
    3. Entre inviáveis: menor violação agregada de restrições vence (caminho em direção à região viável).
    4. Elimina 100% a dependência de pesos arbitrários de penalidade (sem falsos ótimos!).
    """
    f = OBJECTIVES[objetivo](sol)
    v = violacao_total(sol)
    if v <= 1e-9:
        return (0, float(f), 0.0)
    else:
        return (1, float(v), float(f))


def eh_melhor_deb(sol_a, sol_b, objetivo, tol=1e-9):
    """
    Retorna True se sol_a for estritamente superior a sol_b segundo as Regras de Deb (2000).
    """
    fit_a = fitness_deb(sol_a, objetivo)
    fit_b = fitness_deb(sol_b, objetivo)

    # 1. Prioridade de factibilidade: viável vence inviável
    if fit_a[0] < fit_b[0]:
        return True
    if fit_a[0] > fit_b[0]:
        return False

    # 2. Ambas viáveis: menor objetivo puro vence
    if fit_a[0] == 0:
        return fit_a[1] < fit_b[1] - tol

    # 3. Ambas inviáveis: menor violação total de restrições vence
    if fit_a[1] < fit_b[1] - tol:
        return True
    elif fit_a[1] > fit_b[1] + tol:
        return False
    else:
        # Desempate por valor do objetivo
        return fit_a[2] < fit_b[2] - tol


def avaliar(sol, objetivo):
    """
    Retorna o valor escalar representativo segundo o Princípio de Deb:
    - Se viável: f_objetivo(sol)
    - Se inviável: OFFSET_INVIAVEL (1e9) + violacao_total(sol)
    Garante que qualquer solução factível tem avaliação menor que qualquer infactível.
    """
    f = OBJECTIVES[objetivo](sol)
    v = violacao_total(sol)
    if v <= 1e-9:
        return f
    else:
        return 1.0e9 + v


def resumo_solucao(sol):
    """Retorna um dicionário com as métricas relevantes de uma solução."""
    pen = penalidade_total(sol)
    return dict(
        f1=f1_custo(sol),
        f2=f2_desvio_sio2(sol),
        f3=f3_desvio_al2o3(sol),
        viol_disp=violacao_disponibilidade(sol),
        viol_qual=violacao_qualidade(sol),
        penalidade=pen,
        factivel=is_feasible(sol),
    )


def diagnostico_violacoes(sol, tol=1e-9):
    """
    Retorna diagnóstico estruturado e detalhado de todas as restrições violadas:
    - minerios_violados: lista de minérios que violaram disp_min ou disp_max e a magnitude
    - pilhas_violadas: lista de pilhas que violaram limites de SiO2 ou Al2O3 do grupo e magnitude
    - total_viol_disp: soma total de caminhões fora dos limites
    - total_viol_qual: soma total de desvios percentuais de qualidade fora dos limites
    - factivel: booleano de viabilidade estrita
    """
    uso = {m: 0 for m in MINERALS}
    for p in PILE_IDS:
        for m in sol[p]:
            uso[m] += 1

    minerios_violados = []
    total_viol_disp = 0.0
    for m, dados in MINERALS.items():
        if uso[m] < dados["disp_min"]:
            v = float(dados["disp_min"] - uso[m])
            total_viol_disp += v
            minerios_violados.append({
                "minerio": m,
                "uso": uso[m],
                "limite": dados["disp_min"],
                "tipo": "disp_min (subutilizado)",
                "violacao": v,
            })
        elif uso[m] > dados["disp_max"]:
            v = float(uso[m] - dados["disp_max"])
            total_viol_disp += v
            minerios_violados.append({
                "minerio": m,
                "uso": uso[m],
                "limite": dados["disp_max"],
                "tipo": "disp_max (estouro)",
                "violacao": v,
            })

    pilhas_violadas = []
    total_viol_qual = 0.0
    for p in PILE_IDS:
        grp = pile_group(p)
        b = GROUP_BOUNDS[grp]
        fet, sio2, al2o3 = pile_quality(sol[p])

        if sio2 < b["SiO2_min"] - tol:
            v = float(b["SiO2_min"] - sio2)
            total_viol_qual += v
            pilhas_violadas.append({
                "pilha": p,
                "grupo": grp,
                "param": "SiO2",
                "valor": float(sio2),
                "limite": float(b["SiO2_min"]),
                "tipo": "min",
                "violacao": v,
            })
        elif sio2 > b["SiO2_max"] + tol:
            v = float(sio2 - b["SiO2_max"])
            total_viol_qual += v
            pilhas_violadas.append({
                "pilha": p,
                "grupo": grp,
                "param": "SiO2",
                "valor": float(sio2),
                "limite": float(b["SiO2_max"]),
                "tipo": "max",
                "violacao": v,
            })

        if al2o3 < b["Al2O3_min"] - tol:
            v = float(b["Al2O3_min"] - al2o3)
            total_viol_qual += v
            pilhas_violadas.append({
                "pilha": p,
                "grupo": grp,
                "param": "Al2O3",
                "valor": float(al2o3),
                "limite": float(b["Al2O3_min"]),
                "tipo": "min",
                "violacao": v,
            })
        elif al2o3 > b["Al2O3_max"] + tol:
            v = float(al2o3 - b["Al2O3_max"])
            total_viol_qual += v
            pilhas_violadas.append({
                "pilha": p,
                "grupo": grp,
                "param": "Al2O3",
                "valor": float(al2o3),
                "limite": float(b["Al2O3_max"]),
                "tipo": "max",
                "violacao": v,
            })

    factivel = (total_viol_disp <= tol) and (total_viol_qual <= tol)
    return {
        "factivel": factivel,
        "total_viol_disp": total_viol_disp,
        "total_viol_qual": total_viol_qual,
        "minerios_violados": minerios_violados,
        "pilhas_violadas": pilhas_violadas,
    }


def analise_proximidade_alvos(sol, tol_excelente=0.10, tol_bom=0.25):
    """
    Analisa em detalhes a proximidade dos teores de SiO2 e Al2O3 em relação aos alvos
    específicos de cada grupo para todas as 10 pilhas.

    Critérios de Proximidade (definidos e explícitos):
    - 'Excelente' (|desvio| <= 0.10%): teor muito próximo ao alvo operacional
    - 'Bom' (0.10% < |desvio| <= 0.25%): variação pequena e controlada
    - 'Afastado' (|desvio| > 0.25%): variação sensível
    - 'Inviável': ultrapassou os limites min/max regulatórios do grupo
    """
    detalhes_pilhas = {}
    desvios_sio2 = []
    desvios_al2o3 = []
    desvios_sio2_quad = []
    desvios_al2o3_quad = []

    for p in PILE_IDS:
        grp = pile_group(p)
        b = GROUP_BOUNDS[grp]
        fet, sio2, al2o3 = pile_quality(sol[p])

        # SiO2
        alvo_sio2 = b["SiO2_alvo"]
        desvio_sio2 = float(sio2 - alvo_sio2)
        abs_desvio_sio2 = abs(desvio_sio2)
        sio2_quad = float(desvio_sio2 ** 2)
        inviavel_sio2 = (sio2 < b["SiO2_min"] - 1e-9) or (sio2 > b["SiO2_max"] + 1e-9)

        if inviavel_sio2:
            class_sio2 = "Inviável"
        elif abs_desvio_sio2 <= tol_excelente:
            class_sio2 = "Excelente"
        elif abs_desvio_sio2 <= tol_bom:
            class_sio2 = "Bom"
        else:
            class_sio2 = "Afastado"

        # Al2O3
        alvo_al2o3 = b["Al2O3_alvo"]
        desvio_al2o3 = float(al2o3 - alvo_al2o3)
        abs_desvio_al2o3 = abs(desvio_al2o3)
        al2o3_quad = float(desvio_al2o3 ** 2)
        inviavel_al2o3 = (al2o3 < b["Al2O3_min"] - 1e-9) or (al2o3 > b["Al2O3_max"] + 1e-9)

        if inviavel_al2o3:
            class_al2o3 = "Inviável"
        elif abs_desvio_al2o3 <= tol_excelente:
            class_al2o3 = "Excelente"
        elif abs_desvio_al2o3 <= tol_bom:
            class_al2o3 = "Bom"
        else:
            class_al2o3 = "Afastado"

        detalhes_pilhas[p] = {
            "grupo": grp,
            "FeT": float(fet),
            "SiO2": {
                "teor": float(sio2),
                "alvo": float(alvo_sio2),
                "limites": (float(b["SiO2_min"]), float(b["SiO2_max"])),
                "desvio": desvio_sio2,
                "abs_desvio": abs_desvio_sio2,
                "desvio_quad": sio2_quad,
                "classificacao": class_sio2,
            },
            "Al2O3": {
                "teor": float(al2o3),
                "alvo": float(alvo_al2o3),
                "limites": (float(b["Al2O3_min"]), float(b["Al2O3_max"])),
                "desvio": desvio_al2o3,
                "abs_desvio": abs_desvio_al2o3,
                "desvio_quad": al2o3_quad,
                "classificacao": class_al2o3,
            },
        }
        desvios_sio2.append(abs_desvio_sio2)
        desvios_al2o3.append(abs_desvio_al2o3)
        desvios_sio2_quad.append(sio2_quad)
        desvios_al2o3_quad.append(al2o3_quad)

    n_p = float(len(PILE_IDS))
    mean_sio2 = sum(desvios_sio2) / n_p
    mean_al2o3 = sum(desvios_al2o3) / n_p

    var_sio2 = sum((d - mean_sio2) ** 2 for d in desvios_sio2) / n_p
    var_al2o3 = sum((d - mean_al2o3) ** 2 for d in desvios_al2o3) / n_p

    estatisticas = {
        "SiO2": {
            "soma_quadratica_f2": float(sum(desvios_sio2_quad)),
            "desvio_abs_medio": float(mean_sio2),
            "desvio_abs_max": float(max(desvios_sio2)),
            "desvio_abs_min": float(min(desvios_sio2)),
            "desvio_padrao_entre_pilhas": float(var_sio2 ** 0.5),
            "contagem_classificacao": {
                c: sum(1 for p in detalhes_pilhas.values() if p["SiO2"]["classificacao"] == c)
                for c in ["Excelente", "Bom", "Afastado", "Inviável"]
            },
        },
        "Al2O3": {
            "soma_quadratica_f3": float(sum(desvios_al2o3_quad)),
            "desvio_abs_medio": float(mean_al2o3),
            "desvio_abs_max": float(max(desvios_al2o3)),
            "desvio_abs_min": float(min(desvios_al2o3)),
            "desvio_padrao_entre_pilhas": float(var_al2o3 ** 0.5),
            "contagem_classificacao": {
                c: sum(1 for p in detalhes_pilhas.values() if p["Al2O3"]["classificacao"] == c)
                for c in ["Excelente", "Bom", "Afastado", "Inviável"]
            },
        },
    }

    return {
        "pilhas": detalhes_pilhas,
        "estatisticas": estatisticas,
        "criterio_adotado": {
            "excelente_limite_abs": tol_excelente,
            "bom_limite_abs": tol_bom,
            "descricao": "Excelente <= 0.10%, Bom <= 0.25%, Afastado > 0.25%",
        },
    }
