# -*- coding: utf-8 -*-
"""
data.py
Parâmetros do problema 

Todas as massas são tratadas em toneladas (t) internamente, para que o
custo total seja expresso diretamente em R$.
"""

TRUCK_MASS_T = 2000.0  # capacidade fixa do caminhão: 2 kt = 2000 t

# -----------------------------------------------------------------------
# Minérios: preco (R$/t), FeT (%), SiO2 (%), Al2O3 (%), disp_min (caminhões),
# disp_max (caminhões), elegível Sinter 1 (bool), elegível Sinter 2 (bool)
# -----------------------------------------------------------------------
MINERALS = {
    "M1":  dict(preco=320, FeT=66, SiO2=5.0, Al2O3=1.0, disp_min=0, disp_max=6,  s1=True,  s2=True),
    "M2":  dict(preco=280, FeT=64, SiO2=3.0, Al2O3=3.0, disp_min=0, disp_max=8,  s1=True,  s2=True),
    "M3":  dict(preco=270, FeT=62, SiO2=9.0, Al2O3=0.6, disp_min=0, disp_max=8,  s1=True,  s2=True),
    "M4":  dict(preco=190, FeT=58, SiO2=8.0, Al2O3=2.8, disp_min=2, disp_max=12, s1=True,  s2=True),
    "M5":  dict(preco=230, FeT=60, SiO2=6.0, Al2O3=1.5, disp_min=0, disp_max=10, s1=True,  s2=True),
    "M6":  dict(preco=300, FeT=63, SiO2=5.5, Al2O3=1.2, disp_min=0, disp_max=8,  s1=True,  s2=True),
    "M7":  dict(preco=260, FeT=65, SiO2=5.5, Al2O3=1.1, disp_min=0, disp_max=8,  s1=True,  s2=True),
    "M8":  dict(preco=240, FeT=61, SiO2=6.2, Al2O3=1.6, disp_min=0, disp_max=8,  s1=True,  s2=False),
    "M9":  dict(preco=210, FeT=59, SiO2=6.5, Al2O3=1.8, disp_min=0, disp_max=8,  s1=True,  s2=False),
    "M10": dict(preco=180, FeT=57, SiO2=7.0, Al2O3=2.5, disp_min=2, disp_max=10, s1=True,  s2=False),
    "M11": dict(preco=310, FeT=66, SiO2=4.7, Al2O3=0.7, disp_min=1, disp_max=6,  s1=True,  s2=False),
    "M12": dict(preco=200, FeT=60, SiO2=5.8, Al2O3=1.4, disp_min=0, disp_max=8,  s1=True,  s2=False),
    "M13": dict(preco=160, FeT=56, SiO2=9.0, Al2O3=3.2, disp_min=0, disp_max=10, s1=True,  s2=True),
    "M14": dict(preco=150, FeT=56, SiO2=6.0, Al2O3=1.3, disp_min=0, disp_max=6,  s1=True,  s2=True),
    "M15": dict(preco=170, FeT=58, SiO2=8.0, Al2O3=3.5, disp_min=0, disp_max=8,  s1=True,  s2=True),
    "M16": dict(preco=220, FeT=59, SiO2=6.0, Al2O3=1.5, disp_min=0, disp_max=8,  s1=False, s2=True),
    "M17": dict(preco=140, FeT=56, SiO2=8.5, Al2O3=3.6, disp_min=0, disp_max=10, s1=False, s2=True),
    "M18": dict(preco=260, FeT=62, SiO2=5.0, Al2O3=1.0, disp_min=0, disp_max=8,  s1=False, s2=True),
    "M19": dict(preco=190, FeT=57, SiO2=6.3, Al2O3=1.4, disp_min=0, disp_max=8,  s1=False, s2=True),
    "M20": dict(preco=130, FeT=56, SiO2=3.8, Al2O3=3.8, disp_min=1, disp_max=8,  s1=False, s2=True),
}

MINERAL_IDS = list(MINERALS.keys())

# -----------------------------------------------------------------------
# Limites de qualidade por grupo (equipamento)
# -----------------------------------------------------------------------
GROUP_BOUNDS = {
    "S1": dict(FeT_min=0, FeT_max=100,
               SiO2_min=5.0, SiO2_max=6.2, SiO2_alvo=5.55,
               Al2O3_min=1.0, Al2O3_max=2.3, Al2O3_alvo=1.25),
    "S2": dict(FeT_min=0, FeT_max=100,
               SiO2_min=5.0, SiO2_max=6.5, SiO2_alvo=5.50,
               Al2O3_min=1.0, Al2O3_max=2.2, Al2O3_alvo=1.5),
}

# -----------------------------------------------------------------------
# Pilhas: id -> (grupo/equipamento, massa em kt)
# Todas as massas são múltiplos de 2 kt (capacidade do caminhão), logo o
# número de caminhões (posições do vetor de composição) é massa_kt / 2.
# -----------------------------------------------------------------------
PILE_MASS_KT = {
    "P1": ("S1", 20), "P2": ("S1", 22), "P3": ("S1", 24), "P4": ("S1", 26), "P5": ("S1", 28),
    "P6": ("S2", 30), "P7": ("S2", 32), "P8": ("S2", 34), "P9": ("S2", 36), "P10": ("S2", 38),
}

PILES = {}
for pid, (grp, mass_kt) in PILE_MASS_KT.items():
    n_trucks = int(round((mass_kt * 1000.0) / TRUCK_MASS_T))  # kt->t / t_per_truck
    PILES[pid] = dict(grupo=grp, massa_t=mass_kt * 1000.0, n_trucks=n_trucks)

PILE_IDS = list(PILES.keys())


def eligible_minerals(grupo):
    """Retorna a lista de minérios elegíveis para o grupo/equipamento informado ('S1' ou 'S2')."""
    key = "s1" if grupo == "S1" else "s2"
    return [m for m in MINERAL_IDS if MINERALS[m][key]]


ELIGIBLE = {"S1": eligible_minerals("S1"), "S2": eligible_minerals("S2")}

if __name__ == "__main__":
    print("Pilhas:", PILES)
    print("Elegiveis S1 (%d):" % len(ELIGIBLE["S1"]), ELIGIBLE["S1"])
    print("Elegiveis S2 (%d):" % len(ELIGIBLE["S2"]), ELIGIBLE["S2"])
    print("Total trucks S1:", sum(PILES[p]["n_trucks"] for p in PILES if PILES[p]["grupo"] == "S1"))
    print("Total trucks S2:", sum(PILES[p]["n_trucks"] for p in PILES if PILES[p]["grupo"] == "S2"))
