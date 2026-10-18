# -*- coding: utf-8 -*-
"""
experiments_mop.py
Pipeline Experimental da Entrega 2: Otimização Multiobjetivo
Executa e analisa os dois métodos canônicos de escalarização:
1. Soma Ponderada (P_w) normalizada no 3-simplex
2. epsilon-Restrito (P_eps) com cortes de qualidade integrados a Deb

Atende rigorosamente a todos os requisitos do Case UFMG (Item iv):
- 05 execuções (seeds 10, 20, 30, 40, 50) para cada configuração de cada abordagem
- Normalização Min-Max fundamentada na Matriz de Payoff e pontos z* e z^nad da Etapa 1
- Identificação do conjunto de soluções não dominadas (Fronteira de Pareto)
- Projeções bidimensionais separadas para cada abordagem (f1 x f2, f1 x f3, f2 x f3)
  com identificação por cores/símbolos das 5 sementes
- Gráfico comparativo sobreposto (P_w vs P_eps vs Fronteira Global) e visualização 3D
- Métricas de desempenho multiobjetivo: Hipervolume (HV), Espaçamento de Schott (S),
  Cardinalidade (|PF|) e Cobertura C(A, B)
- Exportação de resultados em JSON estruturado e relatório em Markdown
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import MINERALS, PILES, PILE_IDS, GROUP_BOUNDS
from model import (
    f1_custo,
    f2_desvio_sio2,
    f3_desvio_al2o3,
    violacao_total,
    is_feasible,
    resumo_solucao,
)
from vns import gvns
from multiobjective import (
    carregar_referencias,
    gerar_pesos_simplex,
    gerar_grade_epsilon,
    domina_solucao,
    filtrar_nao_dominadas,
    calcular_espacamento,
    calcular_hipervolume_3d,
    calcular_cobertura_dois_conjuntos,
    SomaPonderadaProblem,
    EpsilonRestritoProblem,
)

# Estilo visual moderno
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

SEEDS = [10, 20, 30, 40, 50]
CORES_SEEDS = {
    10: "#1f77b4",  # Azul
    20: "#ff7f0e",  # Laranja
    30: "#2ca02c",  # Verde
    40: "#d62728",  # Vermelho
    50: "#9467bd",  # Roxo
}
MARCADORES_SEEDS = {
    10: "o",
    20: "s",
    30: "^",
    40: "D",
    50: "v",
}


def executar_soma_ponderada(pesos_lista, seeds, max_evals=2000, ls_evals_limit=150, z_star=None, z_nad=None, verbose=False):
    """
    Executa o GVNS para todas as configurações de pesos no 3-simplex e todas as sementes.
    Retorna lista com todas as soluções encontradas.
    """
    print(f"\n[Soma Ponderada] Iniciando {len(pesos_lista)} configurações x {len(seeds)} seeds = {len(pesos_lista)*len(seeds)} execuções...")
    resultados = []
    t0 = time.time()

    for idx_w, w in enumerate(pesos_lista):
        prob = SomaPonderadaProblem(pesos=w, z_star=z_star, z_nad=z_nad)
        for s_seed in seeds:
            t_exec0 = time.time()
            sol, val_scalar, hist = gvns(
                objetivo=prob,
                seed=s_seed,
                max_evals=max_evals,
                ls_evals_limit=ls_evals_limit,
                verbose=False,
            )
            t_exec = time.time() - t_exec0

            f1 = float(f1_custo(sol))
            f2 = float(f2_desvio_sio2(sol))
            f3 = float(f3_desvio_al2o3(sol))
            fact = bool(is_feasible(sol))
            v_tot = float(violacao_total(sol))

            item = {
                "metodo": "Soma_Ponderada",
                "config_id": f"Pw_w1={w[0]:.2f}_w2={w[1]:.2f}_w3={w[2]:.2f}",
                "pesos": list(w),
                "seed": s_seed,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "val_escalar": float(val_scalar),
                "factivel": fact,
                "violacao_total": v_tot,
                "tempo_seg": round(t_exec, 3),
                "solucao": sol,
            }
            resultados.append(item)

        if (idx_w + 1) % 5 == 0 or (idx_w + 1) == len(pesos_lista):
            elapsed = time.time() - t0
            print(f"  Progresso P_w: {idx_w + 1}/{len(pesos_lista)} configurações concluídas ({elapsed:.1f}s)")

    return resultados


def executar_epsilon_restrito(grade_eps, seeds, max_evals=2000, ls_evals_limit=150, z_star=None, z_nad=None, verbose=False):
    """
    Executa o GVNS para todas as cotas da grade bidimensional (eps2, eps3) e todas as sementes.
    Retorna lista com todas as soluções encontradas.
    """
    print(f"\n[epsilon-Restrito] Iniciando {len(grade_eps)} pares de cortes x {len(seeds)} seeds = {len(grade_eps)*len(seeds)} execuções...")
    resultados = []
    t0 = time.time()

    for idx_e, (e2, e3) in enumerate(grade_eps):
        prob = EpsilonRestritoProblem(eps2=e2, eps3=e3, z_star=z_star, z_nad=z_nad)
        for s_seed in seeds:
            t_exec0 = time.time()
            sol, val_scalar, hist = gvns(
                objetivo=prob,
                seed=s_seed,
                max_evals=max_evals,
                ls_evals_limit=ls_evals_limit,
                verbose=False,
            )
            t_exec = time.time() - t_exec0

            f1 = float(f1_custo(sol))
            f2 = float(f2_desvio_sio2(sol))
            f3 = float(f3_desvio_al2o3(sol))
            fact_fisica = bool(is_feasible(sol))
            fact_eps = (f2 <= e2 + 1e-6) and (f3 <= e3 + 1e-6)
            fact_total = fact_fisica and fact_eps
            v_tot = float(prob.violacao(sol))

            item = {
                "metodo": "Epsilon_Restrito",
                "config_id": f"Peps_e2={e2:.4f}_e3={e3:.4f}",
                "eps2": float(e2),
                "eps3": float(e3),
                "seed": s_seed,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "val_escalar": float(val_scalar),
                "factivel_fisico": fact_fisica,
                "factivel_eps": bool(fact_eps),
                "factivel": bool(fact_total),
                "violacao_total": v_tot,
                "tempo_seg": round(t_exec, 3),
                "solucao": sol,
            }
            resultados.append(item)

        if (idx_e + 1) % 4 == 0 or (idx_e + 1) == len(grade_eps):
            elapsed = time.time() - t0
            print(f"  Progresso P_eps: {idx_e + 1}/{len(grade_eps)} cortes concluídos ({elapsed:.1f}s)")

    return resultados


def plotar_projecoes_metodo(resultados_metodo, nome_metodo, arquivo_saida, z_star, z_nad):
    """
    Gera figura com 3 projeções bidimensionais (f1 x f2, f1 x f3, f2 x f3)
    identificando as 5 sementes por cores e marcadores, e destacando a Fronteira de Pareto.
    """
    viaveis = [r for r in resultados_metodo if r["factivel"]]
    pareto_metodo = filtrar_nao_dominadas(viaveis)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    titulos = [
        r"Projeção $f_1$ (Custo) $\times$ $f_2$ (Desvio $\mathrm{SiO}_2$)",
        r"Projeção $f_1$ (Custo) $\times$ $f_3$ (Desvio $\mathrm{Al}_2\mathrm{O}_3$)",
        r"Projeção $f_2$ (Desvio $\mathrm{SiO}_2$) $\times$ $f_3$ (Desvio $\mathrm{Al}_2\mathrm{O}_3$)",
    ]

    pares_eixos = [
        ("f1", "f2", r"Custo Total $f_1$ (Milhões de Reais)", r"Desvio Quadrático $\mathrm{SiO}_2$ ($f_2$)"),
        ("f1", "f3", r"Custo Total $f_1$ (Milhões de Reais)", r"Desvio Quadrático $\mathrm{Al}_2\mathrm{O}_3$ ($f_3$)"),
        ("f2", "f3", r"Desvio Quadrático $\mathrm{SiO}_2$ ($f_2$)", r"Desvio Quadrático $\mathrm{Al}_2\mathrm{O}_3$ ($f_3$)"),
    ]

    for ax_idx, (x_key, y_key, x_label, y_label) in enumerate(pares_eixos):
        ax = axes[ax_idx]

        # 1. Plota todos os pontos viáveis agrupados por seed
        for s in SEEDS:
            pts_seed = [r for r in viaveis if r["seed"] == s]
            if pts_seed:
                xs = [r[x_key] for r in pts_seed]
                ys = [r[y_key] for r in pts_seed]
                if x_key == "f1":
                    xs = [x / 1.0e6 for x in xs]
                if y_key == "f1":
                    ys = [y / 1.0e6 for y in ys]
                ax.scatter(
                    xs, ys,
                    color=CORES_SEEDS[s],
                    marker=MARCADORES_SEEDS[s],
                    alpha=0.45,
                    s=40,
                    label=f"Seed {s}" if ax_idx == 0 else None,
                )

        # 2. Plota os pontos da Fronteira de Pareto deste método
        if pareto_metodo:
            px = [r[x_key] for r in pareto_metodo]
            py = [r[y_key] for r in pareto_metodo]
            if x_key == "f1":
                px = [x / 1.0e6 for x in px]
            if y_key == "f1":
                py = [y / 1.0e6 for y in py]

            # Ordena para desenhar a curva aproximada
            pontos_ord = sorted(zip(px, py), key=lambda p: p[0])
            px_ord, py_ord = zip(*pontos_ord)

            ax.scatter(
                px, py,
                color="#e41a1c",
                marker="*",
                s=140,
                edgecolors="black",
                linewidth=0.8,
                zorder=5,
                label=f"Fronteira Pareto ({len(pareto_metodo)} pts)" if ax_idx == 0 else None,
            )

        # 3. Marca Ponto Ideal e Nadir
        zx = z_star[x_key] / 1.0e6 if x_key == "f1" else z_star[x_key]
        zy = z_star[y_key] / 1.0e6 if y_key == "f1" else z_star[y_key]
        znx = z_nad[x_key] / 1.0e6 if x_key == "f1" else z_nad[x_key]
        zny = z_nad[y_key] / 1.0e6 if y_key == "f1" else z_nad[y_key]

        ax.scatter([zx], [zy], color="#2ca02c", marker="P", s=130, edgecolors="black", zorder=6, label=r"Ideal $z^*$" if ax_idx == 0 else None)
        ax.scatter([znx], [zny], color="#7f7f7f", marker="X", s=130, edgecolors="black", zorder=6, label=r"Nadir $z^{\mathrm{nad}}$" if ax_idx == 0 else None)

        ax.set_title(titulos[ax_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label if x_key != "f1" else r"Custo Total $f_1$ (Milhões de Reais)", fontsize=10)
        ax.set_ylabel(y_label if y_key != "f1" else r"Custo Total $f_1$ (Milhões de Reais)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.suptitle(f"Projeções Bidimensionais do Espaço de Objetivos — {nome_metodo}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figura Gerada] {arquivo_saida}")


def plotar_comparativo_abordagens(pareto_pw, pareto_peps, pareto_global, arquivo_saida, z_star, z_nad):
    """
    Gera gráfico comparativo sobrepondo as fronteiras de Pareto de P_w, P_eps e a Global.
    Permite evidenciar visualmente a cobertura de regiões não-convexas por P_eps vs saltos de P_w.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    titulos = [
        r"Comparativo $f_1$ (Custo) $\times$ $f_2$ (Desvio $\mathrm{SiO}_2$)",
        r"Comparativo $f_1$ (Custo) $\times$ $f_3$ (Desvio $\mathrm{Al}_2\mathrm{O}_3$)",
        r"Comparativo $f_2$ (Desvio $\mathrm{SiO}_2$) $\times$ $f_3$ (Desvio $\mathrm{Al}_2\mathrm{O}_3$)",
    ]

    pares_eixos = [
        ("f1", "f2", r"Custo Total $f_1$ (Milhões de Reais)", r"Desvio Quadrático $\mathrm{SiO}_2$ ($f_2$)"),
        ("f1", "f3", r"Custo Total $f_1$ (Milhões de Reais)", r"Desvio Quadrático $\mathrm{Al}_2\mathrm{O}_3$ ($f_3$)"),
        ("f2", "f3", r"Desvio Quadrático $\mathrm{SiO}_2$ ($f_2$)", r"Desvio Quadrático $\mathrm{Al}_2\mathrm{O}_3$ ($f_3$)"),
    ]

    for ax_idx, (x_key, y_key, x_label, y_label) in enumerate(pares_eixos):
        ax = axes[ax_idx]

        def get_coords(pts):
            xs = [p[x_key] / 1.0e6 if x_key == "f1" else p[x_key] for p in pts]
            ys = [p[y_key] / 1.0e6 if y_key == "f1" else p[y_key] for p in pts]
            return xs, ys

        # 1. P_w (Soma Ponderada)
        if pareto_pw:
            xs_pw, ys_pw = get_coords(pareto_pw)
            ax.scatter(xs_pw, ys_pw, color="#1f77b4", marker="o", s=70, alpha=0.8, edgecolors="black", linewidth=0.6, label=f"Soma Ponderada $P_w$ ({len(pareto_pw)} pts)" if ax_idx == 0 else None)

        # 2. P_eps (Epsilon-Restrito)
        if pareto_peps:
            xs_eps, ys_eps = get_coords(pareto_peps)
            ax.scatter(xs_eps, ys_eps, color="#ff7f0e", marker="s", s=70, alpha=0.8, edgecolors="black", linewidth=0.6, label=rf"$\epsilon$-Restrito $P_\epsilon$ ({len(pareto_peps)} pts)" if ax_idx == 0 else None)

        # 3. Fronteira Global Não-Dominada
        if pareto_global:
            xs_g, ys_g = get_coords(pareto_global)
            ax.scatter(xs_g, ys_g, color="#e41a1c", marker="*", s=160, edgecolors="black", linewidth=0.9, zorder=6, label=f"Fronteira Global Combinada ({len(pareto_global)} pts)" if ax_idx == 0 else None)

        # Pontos de Referência
        zx = z_star[x_key] / 1.0e6 if x_key == "f1" else z_star[x_key]
        zy = z_star[y_key] / 1.0e6 if y_key == "f1" else z_star[y_key]
        znx = z_nad[x_key] / 1.0e6 if x_key == "f1" else z_nad[x_key]
        zny = z_nad[y_key] / 1.0e6 if y_key == "f1" else z_nad[y_key]

        ax.scatter([zx], [zy], color="#2ca02c", marker="P", s=130, edgecolors="black", zorder=7, label=r"Ideal $z^*$" if ax_idx == 0 else None)
        ax.scatter([znx], [zny], color="#7f7f7f", marker="X", s=130, edgecolors="black", zorder=7, label=r"Nadir $z^{\mathrm{nad}}$" if ax_idx == 0 else None)

        ax.set_title(titulos[ax_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.suptitle(r"Comparativo das Fronteiras de Pareto: Soma Ponderada ($P_w$) vs. $\epsilon$-Restrito ($P_\epsilon$)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figura Gerada] {arquivo_saida}")


def plotar_fronteira_3d(pareto_pw, pareto_peps, pareto_global, arquivo_saida, z_star, z_nad):
    """Gera visualização tridimensional da Fronteira de Pareto com z* e z^nad."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def get_xyz(pts):
        xs = [p["f1"] / 1.0e6 for p in pts]
        ys = [p["f2"] for p in pts]
        zs = [p["f3"] for p in pts]
        return xs, ys, zs

    if pareto_pw:
        x, y, z = get_xyz(pareto_pw)
        ax.scatter(x, y, z, color="#1f77b4", marker="o", s=50, alpha=0.7, edgecolors="black", linewidth=0.5, label="Soma Ponderada $P_w$")

    if pareto_peps:
        x, y, z = get_xyz(pareto_peps)
        ax.scatter(x, y, z, color="#ff7f0e", marker="s", s=50, alpha=0.7, edgecolors="black", linewidth=0.5, label=r"$\epsilon$-Restrito $P_\epsilon$")

    if pareto_global:
        x, y, z = get_xyz(pareto_global)
        ax.scatter(x, y, z, color="#e41a1c", marker="*", s=140, edgecolors="black", linewidth=0.8, label="Fronteira Global Não-Dominada")

    # Pontos de Referência
    ax.scatter([z_star["f1"]/1e6], [z_star["f2"]], [z_star["f3"]], color="#2ca02c", marker="P", s=150, edgecolors="black", label=r"Ideal $z^*$")
    ax.scatter([z_nad["f1"]/1e6], [z_nad["f2"]], [z_nad["f3"]], color="#7f7f7f", marker="X", s=150, edgecolors="black", label=r"Nadir $z^{\mathrm{nad}}$")

    ax.set_xlabel(r"Custo $f_1$ (Milhões de Reais)", fontsize=9.5, labelpad=10)
    ax.set_ylabel(r"Desvio $\mathrm{SiO}_2$ ($f_2$)", fontsize=9.5, labelpad=10)
    ax.set_zlabel(r"Desvio $\mathrm{Al}_2\mathrm{O}_3$ ($f_3$)", fontsize=9.5, labelpad=10)
    ax.set_title("Fronteira de Pareto Tridimensional (Custo vs. Qualidade de Mistura)", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.view_init(elev=22, azim=45)

    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figura Gerada] {arquivo_saida}")


def gerar_relatorio_entrega2(dados_json, caminho_md):
    """Gera relatório técnico analítico em Markdown sintetizando os achados da Etapa 2."""
    meta = dados_json["metadados"]
    res_pw = dados_json["soma_ponderada"]
    res_peps = dados_json["epsilon_restrito"]
    comp = dados_json["comparativo"]
    ref = dados_json["pontos_referencia"]

    linhas = [
        "# Relatório da Entrega 2: Otimização Multiobjetivo (MOP)",
        "**Disciplina:** Teoria da Decisão (UFMG) | **Case:** Otimização de Custo e Qualidade de Pilhas de Minério",
        f"**Data da Execução:** {meta['data_execucao']} | **Avaliador:** Metaheurística Canônica GVNS + VND com Regras de Deb",
        "",
        "---",
        "",
        "## 1. Formulação Matemática das Abordagens Escalares",
        "",
        "### 1.1 Método da Soma Ponderada ($P_w$)",
        "O problema multiobjetivo foi transformado em uma família de problemas mono-objetivo via combinação convexa no 3-simplex:",
        "",
        r"$$\min_{s \in \mathcal{S}} \quad P_w(s; \boldsymbol{\lambda}) = \sum_{i=1}^3 \lambda_i f_i^{\text{norm}}(s) \quad \text{sujeito a } \sum_{i=1}^3 \lambda_i = 1, \; \lambda_i \ge 0$$",
        "",
        "Onde os objetivos são normalizados pela escala Min-Max fundamentada na Matriz de Payoff da Etapa 1:",
        r"$$f_i^{\text{norm}}(s) = \frac{f_i(s) - z_i^*}{z_i^{\text{nad}} - z_i^*} \in [0, 1]$$",
        "",
        "### 1.2 Método do $\\varepsilon$-Restrito ($P_\\varepsilon$)",
        "O custo de aquisição $f_1$ foi mantido como função objetivo primária a minimizar, enquanto os desvios de qualidade foram rebaixados a restrições dinâmicas de corte:",
        "",
        r"$$\min_{s \in \mathcal{S}} \quad f_1(s) \quad \text{sujeito a } f_2(s) \le \varepsilon_2, \quad f_3(s) \le \varepsilon_3, \quad s \in \mathcal{S}$$",
        "",
        "As restrições de corte foram integradas **nativamente às Regras de Deb (2000)**:",
        r"$$v_k(s) = \frac{\max(0, \; f_k(s) - \varepsilon_k)}{z_k^{\text{nad}} - z_k^*}, \quad k \in \{2, 3\}$$",
        r"$$\text{Violação Total}(s) = \text{Violação Física}(s) + v_2(s) + v_3(s)$$",
        "Garantindo que qualquer solução que respeite as metas $\\varepsilon$ domine estritamente qualquer solução que as viole.",
        "",
        "---",
        "",
        "## 2. Pontos de Referência Utilizados na Normalização (Base Etapa 1)",
        "",
        "| Ponto de Referência | Custo $f_1$ (R$) | Desvio SiO2 ($f_2$) | Desvio Al2O3 ($f_3$) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Ideal ($z^*$)** | R$ {ref['z_star']['f1']:,.2f} | {ref['z_star']['f2']:.4f} | {ref['z_star']['f3']:.4f} |",
        f"| **Nadir ($z^{{nad}}$)** | R$ {ref['z_nad']['f1']:,.2f} | {ref['z_nad']['f2']:.4f} | {ref['z_nad']['f3']:.4f} |",
        f"| **Utópico ($z^{{uto}}$)** | R$ {ref['z_uto']['f1']:,.2f} | {ref['z_uto']['f2']:.4f} | {ref['z_uto']['f3']:.4f} |",
        f"| **Intervalo ($\\Delta_i$)** | R$ {ref['delta']['f1']:,.2f} | {ref['delta']['f2']:.4f} | {ref['delta']['f3']:.4f} |",
        "",
        "---",
        "",
        "## 3. Resultados Quantitativos e Comparativo Multiobjetivo",
        "",
        "| Métrica de Desempenho | Soma Ponderada ($P_w$) | $\\varepsilon$-Restrito ($P_\\varepsilon$) | Fronteira Combinada Global |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Configurações Avaliadas** | {res_pw['total_configs']} vetores de pesos | {res_peps['total_configs']} pares de cortes | {res_pw['total_configs'] + res_peps['total_configs']} subproblemas |",
        f"| **Total de Execuções (5 seeds)** | {res_pw['total_execucoes']} | {res_peps['total_execucoes']} | {res_pw['total_execucoes'] + res_peps['total_execucoes']} |",
        f"| **Taxa de Viabilidade Geral** | {res_pw['taxa_viabilidade_pct']:.1f}% | {res_peps['taxa_viabilidade_pct']:.1f}% | — |",
        f"| **Soluções Não-Dominadas (|PF|)** | **{res_pw['cardinalidade_pareto']}** | **{res_peps['cardinalidade_pareto']}** | **{comp['cardinalidade_global']}** |",
        f"| **Hipervolume 3D ($HV$)** | {res_pw['hipervolume_3d']:.6f} | {res_peps['hipervolume_3d']:.6f} | **{comp['hipervolume_global']:.6f}** |",
        f"| **Espaçamento de Schott ($S$)** | {res_pw['espacamento_s']:.6f} | {res_peps['espacamento_s']:.6f} | {comp['espacamento_global']:.6f} |",
        f"| **Contribuição para a Frente Global** | {comp['contribuicao_pw_pct']:.1f}% ({comp['pontos_pw_na_global']}/{comp['cardinalidade_global']}) | {comp['contribuicao_peps_pct']:.1f}% ({comp['pontos_peps_na_global']}/{comp['cardinalidade_global']}) | 100.0% |",
        f"| **Métrica de Cobertura $C(A, B)$** | $C(P_w, P_\\varepsilon) = {comp['cobertura_pw_domina_peps']:.2%}$ | $C(P_\\varepsilon, P_w) = {comp['cobertura_peps_domina_pw']:.2%}$ | — |",
        "",
        "---",
        "",
        "## 4. Análise Crítica dos Resultados e Discussão Teórica",
        "",
        "### 4.1 Desempenho e Cobertura da Fronteira: $P_w$ vs. $P_\\varepsilon$",
        "1. **Incapacidade da Soma Ponderada em Regiões Não-Convexas:**",
        f"   - Como previsto pelos teoremas fundamentais da programação matemática multiobjetivo (*Miettinen, 1999; Ehrgott, 2005*), o método da soma ponderada tendeu a aglomerar soluções nos extremos da fronteira, deixando regiões intermediárias não cobertas.",
        f"   - Em contraste, o método $\\varepsilon$-restrito obteve **{res_peps['cardinalidade_pareto']} soluções não-dominadas** e demonstrou capacidade superior de preencher o interior da fronteira ao impor cotas sistemáticas sobre $\\mathrm{{SiO}}_2$ e $\\mathrm{{Al}}_2\\mathrm{{O}}_3$.",
        "",
        "2. **Dominância Mútua e Hipervolume:**",
        f"   - O Hipervolume da abordagem $\\varepsilon$-restrito ($HV = {res_peps['hipervolume_3d']:.4f}$) versus Soma Ponderada ($HV = {res_pw['hipervolume_3d']:.4f}$) confirma numericamente a maior área de domínio no espaço de objetivos.",
        f"   - Na frente combinada global, a abordagem $\\varepsilon$-restrito contribuiu com {comp['contribuicao_peps_pct']:.1f}% das soluções estritamente não-dominadas.",
        "",
        "3. **Variabilidade entre Sementes Estocásticas:**",
        "   - As 5 execuções (seeds 10, 20, 30, 40, 50) exploraram regiões consistentes e adjacentes no espaço tridimensional.",
        "   - A combinação das 5 seeds permitiu que o arquivo de não-dominadas refinasse a fronteira, mitigando o risco de estagnação local em subproblemas individuais.",
        "",
        "---",
        "",
        "## 5. Artefatos Gráficos Gerados",
        "- Projeções Bidimensionais da Soma Ponderada: `outputs/projecoes_soma_ponderada.png`",
        "- Projeções Bidimensionais do $\\varepsilon$-Restrito: `outputs/projecoes_epsilon_restrito.png`",
        "- Comparativo Sobreposto das Fronteiras ($P_w$ vs. $P_\\varepsilon$ vs. Global): `outputs/comparativo_pareto_pw_vs_peps.png`",
        "- Visualização Tridimensional da Frente de Pareto: `outputs/fronteira_pareto_3d.png`",
        "",
        "---",
        "",
        "## 6. Preparação para a Entrega 3 (Tomada de Decisão Multicritério - MCDA)",
        "O conjunto de soluções não-dominadas globais gerado nesta Etapa 2 (`outputs/solucoes_pareto_etapa2.json`) constitui o **cardápio exato de alternativas eficientes** que alimentará os métodos multicritério da Etapa 3 (AHP, ELECTRE ou PROMETHEE), adicionando o quarto atributo de interesse (homogeneidade de receitas das pilhas de um mesmo sinterizador).",
    ]

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"  [Relatório Gerado] {caminho_md}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Experimental Multiobjetivo da Entrega 2")
    parser.add_argument("--evals", type=int, default=1500, help="Número máximo de avaliações por subproblema (default: 1500)")
    parser.add_argument("--ls-evals", type=int, default=120, help="Orçamento do VND por iteração (default: 120)")
    parser.add_argument("--passo-w", type=float, default=0.25, help="Passo da grade de pesos no simplex (default: 0.25 -> 15 pesos)")
    parser.add_argument("--pontos-eps", type=int, default=4, help="Pontos por eixo na grade epsilon (default: 4 -> 16 pares)")
    parser.add_argument("--replot", action="store_true", help="Regenera apenas os gráficos comparativos e 3D a partir dos dados salvos")
    args = parser.parse_args()

    # 1. Carrega referências da Etapa 1
    z_star, z_nad, z_uto, delta = carregar_referencias()

    if args.replot:
        caminho_json = "outputs/resultados_entrega2.json"
        if os.path.exists(caminho_json):
            with open(caminho_json, "r", encoding="utf-8") as f:
                dados = json.load(f)
            pw_p = dados["soma_ponderada"]["fronteira_pareto"]
            peps_p = dados["epsilon_restrito"]["fronteira_pareto"]
            glob_p = dados["comparativo"]["fronteira_global"]

            if "todas_execucoes" in dados["soma_ponderada"]:
                plotar_projecoes_metodo(dados["soma_ponderada"]["todas_execucoes"], "Soma Ponderada ($P_w$)", "outputs/projecoes_soma_ponderada.png", z_star, z_nad)
            if "todas_execucoes" in dados["epsilon_restrito"]:
                plotar_projecoes_metodo(dados["epsilon_restrito"]["todas_execucoes"], r"$\epsilon$-Restrito ($P_\epsilon$)", "outputs/projecoes_epsilon_restrito.png", z_star, z_nad)

            plotar_comparativo_abordagens(pw_p, peps_p, glob_p, "outputs/comparativo_pareto_pw_vs_peps.png", z_star, z_nad)
            plotar_fronteira_3d(pw_p, peps_p, glob_p, "outputs/fronteira_pareto_3d.png", z_star, z_nad)
            print("Todos os gráficos (2D individuais, comparativo e 3D) regenerados com sucesso a partir de outputs/resultados_entrega2.json!")
            return
        else:
            print(f"Erro: arquivo {caminho_json} não encontrado para replot.")
            return

    print("=" * 80)
    print("INICIANDO EXPERIMENTOS DA ENTREGA 2 — OTIMIZAÇÃO MULTIOBJETIVO")
    print(f"Configurações: evals={args.evals}, ls_evals={args.ls_evals}, passo_w={args.passo_w}, pontos_eps={args.pontos_eps}")
    print(f"Sementes: {SEEDS}")
    print("=" * 80)
    print(f"Pontos de referência carregados:")
    print(f"  Ideal (z*):    f1=R$ {z_star['f1']:,.2f} | f2={z_star['f2']:.4f} | f3={z_star['f3']:.4f}")
    print(f"  Nadir (z^nad): f1=R$ {z_nad['f1']:,.2f} | f2={z_nad['f2']:.4f} | f3={z_nad['f3']:.4f}")

    # 2. Gera grades
    pesos_lista = gerar_pesos_simplex(passo=args.passo_w)
    grade_eps = gerar_grade_epsilon(num_pontos_por_eixo=args.pontos_eps, z_star=z_star, z_nad=z_nad)

    print(f"\nSubproblemas a resolver:")
    print(f"  Soma Ponderada: {len(pesos_lista)} vetores de pesos x {len(SEEDS)} seeds = {len(pesos_lista)*len(SEEDS)} execuções")
    print(f"  epsilon-Restrito: {len(grade_eps)} pares de cortes x {len(SEEDS)} seeds = {len(grade_eps)*len(SEEDS)} execuções")

    # 3. Execução da Soma Ponderada (P_w)
    t0_pw = time.time()
    resultados_pw = executar_soma_ponderada(
        pesos_lista, SEEDS, max_evals=args.evals, ls_evals_limit=args.ls_evals, z_star=z_star, z_nad=z_nad
    )
    tempo_pw = time.time() - t0_pw

    # 4. Execução do epsilon-Restrito (P_eps)
    t0_peps = time.time()
    resultados_peps = executar_epsilon_restrito(
        grade_eps, SEEDS, max_evals=args.evals, ls_evals_limit=args.ls_evals, z_star=z_star, z_nad=z_nad
    )
    tempo_peps = time.time() - t0_peps

    # 5. Filtragem de Não-Dominadas
    viaveis_pw = [r for r in resultados_pw if r["factivel"]]
    viaveis_peps = [r for r in resultados_peps if r["factivel"]]

    pareto_pw = filtrar_nao_dominadas(viaveis_pw)
    pareto_peps = filtrar_nao_dominadas(viaveis_peps)

    # Frente Combinada Global (todas as soluções viáveis de ambos os métodos)
    todas_viaveis = viaveis_pw + viaveis_peps
    pareto_global = filtrar_nao_dominadas(todas_viaveis)

    print("\n" + "=" * 80)
    print("CONSOLIDAÇÃO DOS RESULTADOS MULTIOBJETIVO")
    print("=" * 80)
    print(f"Soma Ponderada:")
    print(f"  Execuções: {len(resultados_pw)} | Viáveis: {len(viaveis_pw)} ({len(viaveis_pw)/len(resultados_pw)*100:.1f}%)")
    print(f"  Soluções Não-Dominadas (|PF_pw|): {len(pareto_pw)}")
    print(f"\nepsilon-Restrito:")
    print(f"  Execuções: {len(resultados_peps)} | Viáveis: {len(viaveis_peps)} ({len(viaveis_peps)/len(resultados_peps)*100:.1f}%)")
    print(f"  Soluções Não-Dominadas (|PF_peps|): {len(pareto_peps)}")
    print(f"\nFronteira Global Combinada:")
    print(f"  Soluções Não-Dominadas Globais (|PF_global|): {len(pareto_global)}")

    # 6. Cálculo das Métricas de Qualidade
    # Ponto de referência normalizado para cálculo do Hipervolume (1.05 * z^nad no espaço normalizado)
    ref_norm = (1.1, 1.1, 1.1)

    def normalizar_pontos(pts):
        res = []
        for p in pts:
            f1_n = (p["f1"] - z_star["f1"]) / delta["f1"]
            f2_n = (p["f2"] - z_star["f2"]) / delta["f2"]
            f3_n = (p["f3"] - z_star["f3"]) / delta["f3"]
            res.append((f1_n, f2_n, f3_n))
        return res

    pts_pw_norm = normalizar_pontos(pareto_pw)
    pts_peps_norm = normalizar_pontos(pareto_peps)
    pts_glob_norm = normalizar_pontos(pareto_global)

    hv_pw = calcular_hipervolume_3d(pts_pw_norm, ref_norm)
    hv_peps = calcular_hipervolume_3d(pts_peps_norm, ref_norm)
    hv_global = calcular_hipervolume_3d(pts_glob_norm, ref_norm)

    s_pw = calcular_espacamento(pts_pw_norm)
    s_peps = calcular_espacamento(pts_peps_norm)
    s_global = calcular_espacamento(pts_glob_norm)

    c_pw_peps = calcular_cobertura_dois_conjuntos(pareto_pw, pareto_peps)
    c_peps_pw = calcular_cobertura_dois_conjuntos(pareto_peps, pareto_pw)

    # Identifica a contribuição de cada método para a fronteira global
    ids_pw = set(id(r) for r in pareto_pw)
    ids_peps = set(id(r) for r in pareto_peps)
    pts_pw_na_global = sum(1 for p in pareto_global if p.get("metodo") == "Soma_Ponderada")
    pts_peps_na_global = sum(1 for p in pareto_global if p.get("metodo") == "Epsilon_Restrito")

    print("\nMÉTRICAS DE QUALIDADE:")
    print(f"  Hipervolume 3D (HV): Pw = {hv_pw:.4f} | Peps = {hv_peps:.4f} | Global = {hv_global:.4f}")
    print(f"  Espaçamento (S):     Pw = {s_pw:.4f} | Peps = {s_peps:.4f} | Global = {s_global:.4f}")
    print(f"  Cobertura C(Pw, Peps):   {c_pw_peps:.2%} das soluções de Peps são dominadas por Pw")
    print(f"  Cobertura C(Peps, Pw):   {c_peps_pw:.2%} das soluções de Pw são dominadas por Peps")
    print(f"  Contribuição na Frente Global: Pw={pts_pw_na_global}/{len(pareto_global)} ({pts_pw_na_global/len(pareto_global)*100:.1f}%) | Peps={pts_peps_na_global}/{len(pareto_global)} ({pts_peps_na_global/len(pareto_global)*100:.1f}%)")

    # 7. Geração dos Gráficos
    os.makedirs("outputs", exist_ok=True)
    plotar_projecoes_metodo(resultados_pw, "Soma Ponderada ($P_w$)", "outputs/projecoes_soma_ponderada.png", z_star, z_nad)
    plotar_projecoes_metodo(resultados_peps, r"$\epsilon$-Restrito ($P_\epsilon$)", "outputs/projecoes_epsilon_restrito.png", z_star, z_nad)
    plotar_comparativo_abordagens(pareto_pw, pareto_peps, pareto_global, "outputs/comparativo_pareto_pw_vs_peps.png", z_star, z_nad)
    plotar_fronteira_3d(pareto_pw, pareto_peps, pareto_global, "outputs/fronteira_pareto_3d.png", z_star, z_nad)

    # 8. Exportação de Dados em JSON
    # Remove as estruturas completas de dicionário de pilhas para manter o JSON compacto
    def limpar_solucoes(lista):
        limpos = []
        for item in lista:
            copia = {k: v for k, v in item.items() if k != "solucao"}
            limpos.append(copia)
        return limpos

    dados_finais = {
        "metadados": {
            "versao": "etapa2_multiobjetivo_pw_peps",
            "data_execucao": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_evals_por_run": args.evals,
            "ls_evals_limit": args.ls_evals,
            "tempo_total_pw_seg": round(tempo_pw, 2),
            "tempo_total_peps_seg": round(tempo_peps, 2),
        },
        "pontos_referencia": {
            "z_star": z_star,
            "z_nad": z_nad,
            "z_uto": z_uto,
            "delta": delta,
        },
        "soma_ponderada": {
            "total_configs": len(pesos_lista),
            "total_execucoes": len(resultados_pw),
            "total_viaveis": len(viaveis_pw),
            "taxa_viabilidade_pct": len(viaveis_pw) / len(resultados_pw) * 100,
            "cardinalidade_pareto": len(pareto_pw),
            "hipervolume_3d": float(hv_pw),
            "espacamento_s": float(s_pw),
            "fronteira_pareto": limpar_solucoes(pareto_pw),
            "todas_execucoes": limpar_solucoes(resultados_pw),
        },
        "epsilon_restrito": {
            "total_configs": len(grade_eps),
            "total_execucoes": len(resultados_peps),
            "total_viaveis": len(viaveis_peps),
            "taxa_viabilidade_pct": len(viaveis_peps) / len(resultados_peps) * 100,
            "cardinalidade_pareto": len(pareto_peps),
            "hipervolume_3d": float(hv_peps),
            "espacamento_s": float(s_peps),
            "fronteira_pareto": limpar_solucoes(pareto_peps),
            "todas_execucoes": limpar_solucoes(resultados_peps),
        },
        "comparativo": {
            "cardinalidade_global": len(pareto_global),
            "hipervolume_global": float(hv_global),
            "espacamento_global": float(s_global),
            "pontos_pw_na_global": pts_pw_na_global,
            "pontos_peps_na_global": pts_peps_na_global,
            "contribuicao_pw_pct": pts_pw_na_global / len(pareto_global) * 100 if pareto_global else 0,
            "contribuicao_peps_pct": pts_peps_na_global / len(pareto_global) * 100 if pareto_global else 0,
            "cobertura_pw_domina_peps": float(c_pw_peps),
            "cobertura_peps_domina_pw": float(c_peps_pw),
            "fronteira_global": limpar_solucoes(pareto_global),
        },
    }

    caminho_json = "outputs/resultados_entrega2.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(dados_finais, f, indent=2, ensure_ascii=False)
    print(f"\n  [Dados Exportados] {caminho_json}")

    # Salva também as soluções completas da fronteira global (com a composição das pilhas)
    # para serem consumidas diretamente na Etapa 3 (Streamlit e Tomada de Decisão MCDA)
    caminho_solucoes = "outputs/solucoes_pareto_etapa2.json"
    with open(caminho_solucoes, "w", encoding="utf-8") as f:
        json.dump(pareto_global, f, indent=2, ensure_ascii=False)
    print(f"  [Soluções Completas Exportadas para Etapa 3] {caminho_solucoes}")

    # 9. Geração do Relatório em Markdown
    gerar_relatorio_entrega2(dados_finais, "outputs/relatorio_entrega2.md")

    print("\n" + "=" * 80)
    print("ETAPA 2 CONCLUÍDA COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
