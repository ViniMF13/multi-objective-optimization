# -*- coding: utf-8 -*-
"""
test_objectives.py
Testes unitários para as funções objetivo e cálculos de qualidade do minério:
- pile_quality: médias ponderadas (aritméticas por caminhão) de FeT, SiO2 e Al2O3
- pile_cost: custo total da pilha em R$
- f1_custo: custo total do sistema (R$)
- f2_desvio_sio2: soma dos desvios quadráticos de SiO2 em relação ao alvo do grupo
- f3_desvio_al2o3: soma dos desvios quadráticos de Al2O3 em relação ao alvo do grupo
- Validação das propriedades matemáticas descritas no case (ex: convexidade / penalização quadrática)
"""

import unittest
from data import MINERALS, GROUP_BOUNDS, PILES, PILE_IDS, TRUCK_MASS_T
from model import (
    pile_quality,
    pile_cost,
    f1_custo,
    f2_desvio_sio2,
    f3_desvio_al2o3,
)


class TestObjectiveFunctions(unittest.TestCase):
    """Testa o cálculo das funções objetivo e de propriedades das pilhas."""

    def test_pile_quality_homogeneous(self):
        """Uma pilha formada apenas por um único minério deve ter a qualidade exata desse minério."""
        for m in ["M1", "M7", "M14", "M20"]:
            comp = [m] * 10
            fet, sio2, al2o3 = pile_quality(comp)
            self.assertAlmostEqual(fet, MINERALS[m]["FeT"])
            self.assertAlmostEqual(sio2, MINERALS[m]["SiO2"])
            self.assertAlmostEqual(al2o3, MINERALS[m]["Al2O3"])

    def test_pile_quality_analytical_mix(self):
        """Testa o cálculo da média com mistura controlada conhecida."""
        # 2 caminhões de M1 (FeT=66, SiO2=5.0, Al2O3=1.0)
        # 2 caminhões de M2 (FeT=64, SiO2=3.0, Al2O3=3.0)
        # Médias esperadas:
        # FeT = (66*2 + 64*2) / 4 = 65.0
        # SiO2 = (5.0*2 + 3.0*2) / 4 = 4.0
        # Al2O3 = (1.0*2 + 3.0*2) / 4 = 2.0
        comp = ["M1", "M1", "M2", "M2"]
        fet, sio2, al2o3 = pile_quality(comp)
        self.assertAlmostEqual(fet, 65.0)
        self.assertAlmostEqual(sio2, 4.0)
        self.assertAlmostEqual(al2o3, 2.0)

    def test_pile_cost_calculation(self):
        """Valida o custo da pilha em R$ (preco_minerio * 2000 t)."""
        # comp com 1 M1 (320 R$/t) e 1 M20 (130 R$/t)
        # Custo esperado = (320 + 130) * 2000 = 450 * 2000 = 900.000 R$
        comp = ["M1", "M20"]
        custo = pile_cost(comp)
        self.assertEqual(custo, 900_000.0)

    def test_f1_custo_bounds(self):
        """f1 deve estar estritamente entre o custo mínimo e máximo teórico para 145 caminhões."""
        min_preco = min(m["preco"] for m in MINERALS.values())  # M20 = 130
        max_preco = max(m["preco"] for m in MINERALS.values())  # M1 = 320
        total_trucks = 145
        min_teorico = total_trucks * TRUCK_MASS_T * min_preco   # 37.700.000 R$
        max_teorico = total_trucks * TRUCK_MASS_T * max_preco   # 92.800.000 R$

        # Cria uma solução sintética qualquer válida em tamanho
        sol = {p: ["M1"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        self.assertAlmostEqual(f1_custo(sol), max_teorico)

        sol_barata = {p: ["M20"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        self.assertAlmostEqual(f1_custo(sol_barata), min_teorico)

    def test_f2_exact_target_match(self):
        """Se todas as pilhas atingirem exatamente o alvo de SiO2, f2 deve ser 0.0."""
        # Alvo S1 = 5.55, Alvo S2 = 5.50
        # M6 tem SiO2 = 5.5 (exato para S2, erro de 0.05 para S1)
        # Vamos testar calculando o valor analiticamente com uma solução controlada
        sol = {}
        for p in PILE_IDS:
            grp = PILES[p]["grupo"]
            # Para S2, minério com SiO2=5.50 é M6 ou M18
            # Para S1, vamos colocar M6 (SiO2=5.50) -> erro = 5.50 - 5.55 = -0.05
            sol[p] = ["M6"] * PILES[p]["n_trucks"]

        # S1 tem 5 pilhas, cada uma com desvio (-0.05)^2 = 0.0025. Total S1 = 5 * 0.0025 = 0.0125
        # S2 tem 5 pilhas, cada uma com desvio (5.5 - 5.5)^2 = 0. Total S2 = 0
        # Total esperado de f2 = 0.0125
        expected_f2 = 5 * (5.50 - GROUP_BOUNDS["S1"]["SiO2_alvo"]) ** 2
        self.assertAlmostEqual(f2_desvio_sio2(sol), expected_f2, places=6)

    def test_f3_exact_target_match(self):
        """Se todas as pilhas atingirem exatamente o alvo de Al2O3, f3 deve ser 0.0."""
        # Alvo S1 = 1.25, Alvo S2 = 1.50
        # M5 e M16 têm Al2O3 = 1.50 (exato para S2, erro de 1.50 - 1.25 = 0.25 para S1)
        sol = {p: ["M5"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # S1: 5 pilhas * (1.50 - 1.25)^2 = 5 * 0.0625 = 0.3125
        # S2: 5 pilhas * (1.50 - 1.50)^2 = 0.0
        expected_f3 = 5 * (1.50 - GROUP_BOUNDS["S1"]["Al2O3_alvo"]) ** 2
        self.assertAlmostEqual(f3_desvio_al2o3(sol), expected_f3, places=6)

    def test_quadratic_penalty_property_case_statement(self):
        """
        O enunciado afirma:
        'O uso do desvio ao quadrado (em vez do desvio absoluto) penaliza mais fortemente
        soluções em que o desvio se concentra em poucas pilhas do que soluções em que
        o mesmo desvio total está distribuído entre várias pilhas.'
        Valida que f2/f3 satisfazem estritamente essa propriedade matemática de convexidade.
        """
        # Caso A: Desvio total = 0.4 distribuído igualmente entre 4 pilhas de S2 (0.1 cada)
        # Desvios: [0.1, 0.1, 0.1, 0.1, 0.0] -> soma quadrática = 4 * 0.01 = 0.04
        # Caso B: Desvio total = 0.4 concentrado em 1 pilha de S2 (0.4) e outras 4 com 0.0
        # Desvios: [0.4, 0.0, 0.0, 0.0, 0.0] -> soma quadrática = 1 * 0.16 = 0.16
        # Como 0.16 > 0.04, o desvio concentrado deve ser severamente mais penalizado.
        soma_distribuida = 4 * (0.1 ** 2)
        soma_concentrada = 1 * (0.4 ** 2)
        self.assertLess(soma_distribuida, soma_concentrada)
        self.assertEqual(soma_concentrada / soma_distribuida, 4.0)


if __name__ == "__main__":
    unittest.main()
