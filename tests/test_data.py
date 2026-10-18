# -*- coding: utf-8 -*-
"""
test_data.py
Testes unitários para validação de integridade dos parâmetros do problema (data.py)
em relação às especificações do case (Case-UFMG-Teoria_da_Decisao.pdf).
"""

import unittest
from data import (
    TRUCK_MASS_T,
    MINERALS,
    MINERAL_IDS,
    GROUP_BOUNDS,
    PILE_MASS_KT,
    PILES,
    PILE_IDS,
    ELIGIBLE,
    eligible_minerals,
)


class TestProblemData(unittest.TestCase):
    """Testa a integridade de dados e conformidade com o enunciado do Case."""

    def test_truck_capacity(self):
        """A capacidade de cada caminhão deve ser exatamente 2 kt = 2000 t."""
        self.assertEqual(TRUCK_MASS_T, 2000.0)

    def test_minerals_count_and_keys(self):
        """O problema possui exatamente 20 minérios (M1 a M20)."""
        expected_ids = [f"M{i}" for i in range(1, 21)]
        self.assertEqual(len(MINERALS), 20)
        self.assertEqual(MINERAL_IDS, expected_ids)

    def test_all_minerals_specifications(self):
        """Valida valores exatos de cada minério conforme tabela do enunciado (pág. 7)."""
        expected = {
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
        for mid, specs in expected.items():
            self.assertIn(mid, MINERALS)
            for k, val in specs.items():
                self.assertEqual(MINERALS[mid][k], val, f"Divergência em {mid}[{k}]")

    def test_group_bounds(self):
        """Valida os limites e alvos de qualidade de Sinter 1 e Sinter 2."""
        s1 = GROUP_BOUNDS["S1"]
        self.assertEqual(s1["FeT_min"], 0)
        self.assertEqual(s1["FeT_max"], 100)
        self.assertAlmostEqual(s1["SiO2_min"], 5.0)
        self.assertAlmostEqual(s1["SiO2_max"], 6.2)
        self.assertAlmostEqual(s1["SiO2_alvo"], 5.55)
        self.assertAlmostEqual(s1["Al2O3_min"], 1.0)
        self.assertAlmostEqual(s1["Al2O3_max"], 2.3)
        self.assertAlmostEqual(s1["Al2O3_alvo"], 1.25)

        s2 = GROUP_BOUNDS["S2"]
        self.assertEqual(s2["FeT_min"], 0)
        self.assertEqual(s2["FeT_max"], 100)
        self.assertAlmostEqual(s2["SiO2_min"], 5.0)
        self.assertAlmostEqual(s2["SiO2_max"], 6.5)
        self.assertAlmostEqual(s2["SiO2_alvo"], 5.50)
        self.assertAlmostEqual(s2["Al2O3_min"], 1.0)
        self.assertAlmostEqual(s2["Al2O3_max"], 2.2)
        self.assertAlmostEqual(s2["Al2O3_alvo"], 1.50)

    def test_pile_specifications(self):
        """Valida quantidade de caminhões, massas e grupos das pilhas P1 a P10."""
        expected_piles = {
            "P1": ("S1", 20, 10),
            "P2": ("S1", 22, 11),
            "P3": ("S1", 24, 12),
            "P4": ("S1", 26, 13),
            "P5": ("S1", 28, 14),
            "P6": ("S2", 30, 15),
            "P7": ("S2", 32, 16),
            "P8": ("S2", 34, 17),
            "P9": ("S2", 36, 18),
            "P10": ("S2", 38, 19),
        }
        self.assertEqual(len(PILES), 10)
        for pid, (grp, mass_kt, n_trucks) in expected_piles.items():
            self.assertIn(pid, PILES)
            self.assertEqual(PILES[pid]["grupo"], grp)
            self.assertEqual(PILES[pid]["massa_t"], mass_kt * 1000.0)
            self.assertEqual(PILES[pid]["n_trucks"], n_trucks)

    def test_total_trucks_and_mass(self):
        """Valida total de caminhões necessários para S1, S2 e total geral."""
        s1_trucks = sum(PILES[p]["n_trucks"] for p in PILE_IDS if PILES[p]["grupo"] == "S1")
        s2_trucks = sum(PILES[p]["n_trucks"] for p in PILE_IDS if PILES[p]["grupo"] == "S2")
        total_trucks = sum(PILES[p]["n_trucks"] for p in PILE_IDS)

        self.assertEqual(s1_trucks, 60)
        self.assertEqual(s2_trucks, 85)
        self.assertEqual(total_trucks, 145)

    def test_mineral_eligibility(self):
        """Valida os minérios elegíveis para cada grupo."""
        s1_expected = [f"M{i}" for i in range(1, 16)]
        s2_expected = [f"M{i}" for i in range(1, 8)] + [f"M{i}" for i in range(13, 21)]

        self.assertEqual(sorted(ELIGIBLE["S1"]), sorted(s1_expected))
        self.assertEqual(sorted(ELIGIBLE["S2"]), sorted(s2_expected))
        self.assertEqual(eligible_minerals("S1"), ELIGIBLE["S1"])
        self.assertEqual(eligible_minerals("S2"), ELIGIBLE["S2"])

    def test_capacity_feasibility_bounds(self):
        """A capacidade máxima total de caminhões deve ser >= 145 para ser viável."""
        total_max_trucks = sum(m["disp_max"] for m in MINERALS.values())
        total_min_trucks = sum(m["disp_min"] for m in MINERALS.values())
        self.assertGreaterEqual(total_max_trucks, 145)
        self.assertLessEqual(total_min_trucks, 145)


if __name__ == "__main__":
    unittest.main()
