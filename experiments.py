# -*- coding: utf-8 -*-
"""
experiments.py
Executa o GVNS 5 vezes para cada função objetivo mono-objetivo (f1, f2, f3),
reporta min/std/max do valor final de cada execução, gera as curvas de
convergência sobrepostas e uma figura ilustrando a melhor solução de cada
objetivo.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vns import gvns
from model import (OBJECTIVES, resumo_solucao, pile_quality, pile_group,
                    f1_custo, f2_desvio_sio2, f3_desvio_al2o3)
from data import PILE_IDS, GROUP_BOUNDS, MINERALS

OUT_DIR = "outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

N_RUNS = 5
MAX_EVALS = 25000
SEEDS = [10, 20, 30, 40, 50]

OBJ_LABELS = {
    "f1": "f1 - Custo total (R$)",
    "f2": "f2 - Soma dos desvios quadraticos de SiO2",
    "f3": "f3 - Soma dos desvios quadraticos de Al2O3",
}


def run_all():
    resultados = {}  # obj -> lista de dicts {seed, valor_final(puro), historico, sol, resumo}
    for obj in ["f1", "f2", "f3"]:
        resultados[obj] = []
        for seed in SEEDS:
            sol, valor_penalizado, historico = gvns(obj, seed=seed, max_evals=MAX_EVALS)
            resumo = resumo_solucao(sol)
            valor_puro = resumo[obj]  # valor do objetivo sem penalidade, na melhor solução
            resultados[obj].append(dict(
                seed=seed,
                valor_puro=valor_puro,
                valor_penalizado=valor_penalizado,
                historico=historico,
                sol=sol,
                resumo=resumo,
            ))
            print(f"[{obj}] seed={seed} -> valor={valor_puro:.4f} factivel={resumo['factivel']}")
    return resultados


def tabela_estatisticas(resultados):
    linhas = []
    for obj in ["f1", "f2", "f3"]:
        vals = [r["valor_puro"] for r in resultados[obj]]
        linhas.append(dict(
            objetivo=obj,
            min=float(np.min(vals)),
            max=float(np.max(vals)),
            std=float(np.std(vals)),
            media=float(np.mean(vals)),
            valores=vals,
        ))
    return linhas


def plot_convergencia(resultados, obj, filename):
    plt.figure(figsize=(7, 4.5))
    for r in resultados[obj]:
        hist = r["historico"]
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        plt.plot(xs, ys, label=f"seed={r['seed']}")
    plt.xlabel("Número de avaliações de soluções candidatas")
    plt.ylabel("Melhor valor encontrado (fitness penalizado)")
    plt.title(f"Curvas de convergência - {OBJ_LABELS[obj]}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_melhor_solucao(resultados, obj, filename):
    """Ilustra a melhor solução encontrada para `obj`: qualidade (SiO2 e
    Al2O3) de cada pilha comparada aos limites/alvo do respectivo grupo."""
    # escolhe a execução com melhor (menor) valor_puro
    melhor_r = min(resultados[obj], key=lambda r: r["valor_puro"])
    sol = melhor_r["sol"]

    sio2_vals, al2o3_vals, cores = [], [], []
    sio2_min, sio2_max, sio2_alvo = [], [], []
    al_min, al_max, al_alvo = [], [], []
    for p in PILE_IDS:
        grp = pile_group(p)
        _, sio2, al2o3 = pile_quality(sol[p])
        b = GROUP_BOUNDS[grp]
        sio2_vals.append(sio2); al2o3_vals.append(al2o3)
        sio2_min.append(b["SiO2_min"]); sio2_max.append(b["SiO2_max"]); sio2_alvo.append(b["SiO2_alvo"])
        al_min.append(b["Al2O3_min"]); al_max.append(b["Al2O3_max"]); al_alvo.append(b["Al2O3_alvo"])
        cores.append("tab:blue" if grp == "S1" else "tab:orange")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(PILE_IDS))

    ax = axes[0]
    ax.bar(x, sio2_vals, color=cores)
    ax.plot(x, sio2_alvo, "k--", marker="o", label="alvo do grupo")
    ax.plot(x, sio2_min, "r:", label="min/max do grupo")
    ax.plot(x, sio2_max, "r:")
    ax.set_xticks(x); ax.set_xticklabels(PILE_IDS, rotation=45)
    ax.set_ylabel("SiO2 (%)")
    ax.set_title("SiO2 por pilha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(x, al2o3_vals, color=cores)
    ax.plot(x, al_alvo, "k--", marker="o", label="alvo do grupo")
    ax.plot(x, al_min, "r:", label="min/max do grupo")
    ax.plot(x, al_max, "r:")
    ax.set_xticks(x); ax.set_xticklabels(PILE_IDS, rotation=45)
    ax.set_ylabel("Al2O3 (%)")
    ax.set_title("Al2O3 por pilha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Melhor solucao encontrada para {OBJ_LABELS[obj]} "
                 f"(azul = Sinter 1, laranja = Sinter 2)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return melhor_r


def salvar_resumo_json(resultados, filename):
    dump = {}
    for obj in resultados:
        dump[obj] = [dict(seed=r["seed"], valor_puro=r["valor_puro"],
                           resumo=r["resumo"]) for r in resultados[obj]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    resultados = run_all()

    stats = tabela_estatisticas(resultados)
    print("\n=== Estatisticas (min/std/max) das 5 execucoes por objetivo ===")
    for s in stats:
        print(f"{s['objetivo']}: min={s['min']:.4f}  max={s['max']:.4f}  "
              f"std={s['std']:.4f}  media={s['media']:.4f}")

    for obj in ["f1", "f2", "f3"]:
        plot_convergencia(resultados, obj, os.path.join(OUT_DIR, f"convergencia_{obj}.png"))
        melhor = plot_melhor_solucao(resultados, obj, os.path.join(OUT_DIR, f"melhor_solucao_{obj}.png"))
        print(f"Melhor execucao para {obj}: seed={melhor['seed']} valor={melhor['valor_puro']:.4f} "
              f"resumo={melhor['resumo']}")

    salvar_resumo_json(resultados, os.path.join(OUT_DIR, "resultados_entrega1.json"))
    print("\nArquivos de saida salvos em", OUT_DIR)
