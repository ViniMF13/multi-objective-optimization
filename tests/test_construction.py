# -*- coding: utf-8 -*-
"""
test_construction.py
Testes unitários para a heurística construtiva (construction.py).
Garante que a solução inicial é construída conforme as regras do problema e do enunciado:
- Atendimento exato da massa (caminhões por pilha)
- Respeito estrito à elegibilidade de equipamentos
- Priorização de disponibilidade mínima
- Comportamento em relação a limites de qualidade e disponibilidade máxima
"""

import unittest
from construction import construir_solucao
from data import PILES, PILE_IDS, ELIGIBLE, MINERALS
from model import (
    violacao_disponibilidade,
    violacao_qualidade,
    is_feasible,
    resumo_solucao,
)


class TestConstructionHeuristic(unittest.TestCase):
    """Testes da heurística construtiva."""

    def setUp(self):
        self.sol = construir_solucao(seed=42)

    def test_solution_structure(self):
        """A solução deve conter todas as 10 pilhas e cada pilha o número exato de caminhões."""
        self.assertEqual(set(self.sol.keys()), set(PILE_IDS))
        total_trucks = 0
        for p in PILE_IDS:
            expected_n = PILES[p]["n_trucks"]
            actual_n = len(self.sol[p])
            self.assertEqual(
                actual_n,
                expected_n,
                f"Pilha {p} possui {actual_n} caminhões, esperado {expected_n}",
            )
            total_trucks += actual_n
        self.assertEqual(total_trucks, 145)

    def test_equipment_eligibility_s1(self):
        """Pilhas de Sinter 1 (P1 a P5) só devem conter minérios elegíveis para S1."""
        s1_eligible = set(ELIGIBLE["S1"])
        for p in ["P1", "P2", "P3", "P4", "P5"]:
            for m in self.sol[p]:
                self.assertIn(
                    m,
                    s1_eligible,
                    f"Minério {m} alocado na pilha {p} não é elegível para S1",
                )
                # Minérios exclusivos de S2 (M16-M20) não podem estar em S1
                self.assertNotIn(m, {"M16", "M17", "M18", "M19", "M20"})

    def test_equipment_eligibility_s2(self):
        """Pilhas de Sinter 2 (P6 a P10) só devem conter minérios elegíveis para S2."""
        s2_eligible = set(ELIGIBLE["S2"])
        for p in ["P6", "P7", "P8", "P9", "P10"]:
            for m in self.sol[p]:
                self.assertIn(
                    m,
                    s2_eligible,
                    f"Minério {m} alocado na pilha {p} não é elegível para S2",
                )
                # Minérios exclusivos de S1 (M8-M12) não podem estar em S2
                self.assertNotIn(m, {"M8", "M9", "M10", "M11", "M12"})

    def test_disp_min_prioritization(self):
        """A heurística deve garantir que minérios com disp_min > 0 tenham a cota mínima atingida."""
        uso = {m: 0 for m in MINERALS}
        for p in PILE_IDS:
            for m in self.sol[p]:
                uso[m] += 1

        min_demands = {m: data["disp_min"] for m, data in MINERALS.items() if data["disp_min"] > 0}
        self.assertEqual(min_demands, {"M4": 2, "M10": 2, "M11": 1, "M20": 1})

        for m, req in min_demands.items():
            self.assertGreaterEqual(
                uso[m],
                req,
                f"Minério {m} com disp_min={req} foi alocado apenas {uso[m]} vezes",
            )

    def test_disp_max_compliance(self):
        """Verifica se a heurística respeita ou estoura disp_max e mede a violação."""
        uso = {m: 0 for m in MINERALS}
        for p in PILE_IDS:
            for m in self.sol[p]:
                uso[m] += 1

        viol_max = sum(max(0, uso[m] - MINERALS[m]["disp_max"]) for m in MINERALS)
        viol_disp_func = violacao_disponibilidade(self.sol)
        # Como disp_min foi satisfeito, a violação total de disponibilidade deve ser exatamente o estouro de disp_max
        self.assertEqual(viol_disp_func, viol_max)

    def test_greedy_pricing_preference(self):
        """A heurística deve priorizar os minérios mais baratos após alocar disp_min."""
        # M20 (R$ 130) e M17 (R$ 140) são os mais baratos de S2
        # M14 (R$ 150) e M13 (R$ 160) são os mais baratos de S1
        # Eles devem ter alta presença na solução inicial
        uso = {m: 0 for m in MINERALS}
        for p in PILE_IDS:
            for m in self.sol[p]:
                uso[m] += 1

        # M14 (disp_max=6) deve ter sido totalmente utilizado em S1
        self.assertEqual(uso["M14"], MINERALS["M14"]["disp_max"])
        # M20 (disp_max=8) deve ter sido totalmente utilizado em S2
        self.assertEqual(uso["M20"], MINERALS["M20"]["disp_max"])

    def test_initial_solution_valid_structure(self):
        """Valida que o resumo_solucao da solução inicial pode ser gerado sem erros."""
        resumo = resumo_solucao(self.sol)
        self.assertIn("f1", resumo)
        self.assertIn("f2", resumo)
        self.assertIn("f3", resumo)
        self.assertIn("viol_disp", resumo)
        self.assertIn("viol_qual", resumo)
        self.assertIn("factivel", resumo)
        self.assertIsInstance(resumo["f1"], (int, float))
        self.assertGreater(resumo["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
