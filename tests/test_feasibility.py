# -*- coding: utf-8 -*-
"""
test_feasibility.py
Testes unitários para verificação de viabilidade e cálculo de violações:
- violacao_disponibilidade: limites de disp_min e disp_max de cada minério
- violacao_qualidade: limites min e max de SiO2 e Al2O3 específicos de cada grupo (S1 vs S2)
- is_feasible: função booleana com tolerância numérica
- Testes de fronteira e especificidade de grupo
"""

import unittest
from data import MINERALS, GROUP_BOUNDS, PILES, PILE_IDS
from model import (
    violacao_disponibilidade,
    violacao_qualidade,
    is_feasible,
)


class TestFeasibilityAndConstraints(unittest.TestCase):
    """Testa a avaliação e o cumprimento de todas as restrições do problema."""

    def test_disp_violation_zero_when_compliant(self):
        """Quando o uso de todos os minérios respeita [disp_min, disp_max], a violação deve ser zero."""
        # Cria uma alocação sintética perfeitamente viável em disponibilidade
        # Disponibilidades mínimas: M4 (2), M10 (2), M11 (1), M20 (1) = total 6
        # Distribuímos 145 caminhões sem ultrapassar nenhum disp_max
        sol = {}
        trucks_needed = {p: PILES[p]["n_trucks"] for p in PILE_IDS}
        # Montamos uma lista de minérios viáveis
        # S1 precisa de 60 caminhões, S2 precisa de 85 caminhões
        s1_pool = []
        for m in ["M4", "M10", "M11", "M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9", "M12", "M13", "M14"]:
            qtd = min(MINERALS[m]["disp_max"], 60 - len(s1_pool))
            s1_pool.extend([m] * qtd)
            if len(s1_pool) == 60:
                break

        s2_pool = ["M20"]  # disp_min = 1
        for m in ["M15", "M16", "M17", "M18", "M19", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M13"]:
            # Garante que não estouramos o disp_max somado com o que já foi usado em s1_pool
            usado_s1 = s1_pool.count(m)
            sobra = MINERALS[m]["disp_max"] - usado_s1
            if sobra > 0:
                qtd = min(sobra, 85 - len(s2_pool))
                s2_pool.extend([m] * qtd)
            if len(s2_pool) == 85:
                break

        # Distribui nas pilhas de S1
        idx1 = 0
        for p in ["P1", "P2", "P3", "P4", "P5"]:
            n = PILES[p]["n_trucks"]
            sol[p] = s1_pool[idx1 : idx1 + n]
            idx1 += n

        # Distribui nas pilhas de S2
        idx2 = 0
        for p in ["P6", "P7", "P8", "P9", "P10"]:
            n = PILES[p]["n_trucks"]
            sol[p] = s2_pool[idx2 : idx2 + n]
            idx2 += n

        self.assertEqual(violacao_disponibilidade(sol), 0.0)

    def test_disp_violation_under_min(self):
        """Se um minério com disp_min > 0 for subutilizado, a violação deve ser a diferença exata."""
        # Cria uma solução sintética onde M4 (disp_min=2) é usado 0 vezes
        # e M10 (disp_min=2) é usado 1 vez.
        # Demais minérios com disp_min=0 não violam mínimo
        sol = {p: ["M1"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M4 disp_min=2 (usado 0 -> viol 2)
        # M10 disp_min=2 (usado 0 -> viol 2)
        # M11 disp_min=1 (usado 0 -> viol 1)
        # M20 disp_min=1 (usado 0 -> viol 1)
        # M1 usado 145 vezes (disp_max=6 -> estouro 139)
        # Violação esperada de disponibilidade: 2 + 2 + 1 + 1 + 139 = 145
        viol = violacao_disponibilidade(sol)
        self.assertEqual(viol, 145.0)

    def test_quality_violation_group_specificity(self):
        """
        Testa os limites de qualidade estritos de cada grupo:
        - S1: SiO2 in [5.0, 6.2], Al2O3 in [1.0, 2.3]
        - S2: SiO2 in [5.0, 6.5], Al2O3 in [1.0, 2.2]
        Um teor de SiO2 de 6.4% é VIÁVEL para S2, mas INVIÁVEL para S1 (violação 0.2).
        Um teor de Al2O3 de 2.25% é VIÁVEL para S1, mas INVIÁVEL para S2 (violação 0.05).
        """
        # Pilhas homogêneas com minérios específicos
        # M9 tem SiO2 = 6.5, Al2O3 = 1.8
        # Se colocado em S1: SiO2 excede max (6.5 - 6.2 = 0.3 de violação por pilha)
        sol_s1_viola = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M7 tem SiO2=5.5 (dentro), Al2O3=1.1 (dentro de ambos) -> violação 0
        self.assertEqual(violacao_qualidade(sol_s1_viola), 0.0)

        # Agora coloca M9 (SiO2=6.5) apenas na pilha P1 (S1)
        sol_s1_viola["P1"] = ["M9"] * PILES["P1"]["n_trucks"]
        # P1 está em S1 (max SiO2 = 6.2). SiO2 de M9 = 6.5 -> violação de 0.3
        self.assertAlmostEqual(violacao_qualidade(sol_s1_viola), 0.3, places=6)

        # Se colocar M9 em P6 (S2): max SiO2 de S2 é 6.5. Violação deve ser ZERO!
        sol_s2_ok = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        sol_s2_ok["P6"] = ["M9"] * PILES["P6"]["n_trucks"]
        self.assertAlmostEqual(violacao_qualidade(sol_s2_ok), 0.0, places=6)

    def test_quality_violation_lower_bound(self):
        """Testa violação abaixo do limite mínimo de qualidade."""
        # M2 tem SiO2 = 3.0, que é < 5.0 (min de ambos os grupos).
        # Colocado em P1 (S1): violação de SiO2 = 5.0 - 3.0 = 2.0
        # M2 tem Al2O3 = 3.0, que é > 2.3 (max de S1). Violação de Al2O3 = 3.0 - 2.3 = 0.7
        # Violação total de qualidade esperada para P1 = 2.0 + 0.7 = 2.7
        sol = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}  # M7 é perfeitamente viável
        sol["P1"] = ["M2"] * PILES["P1"]["n_trucks"]
        self.assertAlmostEqual(violacao_qualidade(sol), 2.7, places=6)

    def test_is_feasible_function(self):
        """is_feasible só retorna True se ambas as violações forem <= tolerância."""
        sol_viavel_qual = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M7 é viável em qualidade, mas como é o único usado (145 vezes), estoura disp_max=8 -> inviável
        self.assertFalse(is_feasible(sol_viavel_qual))

        # Teste com tolerância
        self.assertTrue(is_feasible(sol_viavel_qual, tol=1000.0))
        self.assertFalse(is_feasible(sol_viavel_qual, tol=1e-9))

    def test_diagnostico_violacoes_structure(self):
        """Testa se diagnostico_violacoes detalha perfeitamente os minérios e pilhas violadas."""
        from model import diagnostico_violacoes
        sol = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M7 é viável em qualidade, mas usado 145 vezes (estoura disp_max=8 por 137 caminhões)
        # Além disso, subutiliza M4 (disp_min=2), M10 (2), M11 (1) e M20 (1)
        diag = diagnostico_violacoes(sol)
        self.assertFalse(diag["factivel"])
        self.assertEqual(diag["total_viol_qual"], 0.0)
        self.assertEqual(diag["total_viol_disp"], 143.0)

        # Minérios com violação
        minerios_viol = {item["minerio"]: item for item in diag["minerios_violados"]}
        self.assertIn("M7", minerios_viol)
        self.assertEqual(minerios_viol["M7"]["violacao"], 137.0)
        self.assertEqual(minerios_viol["M7"]["tipo"], "disp_max (estouro)")
        self.assertIn("M4", minerios_viol)
        self.assertEqual(minerios_viol["M4"]["violacao"], 2.0)

    def test_analise_proximidade_alvos_structure(self):
        """Testa o cálculo detalhado de proximidade de alvos e as faixas de classificação."""
        from model import analise_proximidade_alvos
        sol = {p: ["M6"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M6 tem SiO2 = 5.50 (alvo S2 = 5.50 -> desvio 0.0 'Excelente', alvo S1 = 5.55 -> desvio 0.05 'Excelente')
        # M6 tem Al2O3 = 1.20 (alvo S1 = 1.25 -> desvio 0.05 'Excelente', alvo S2 = 1.50 -> desvio 0.30 'Afastado')
        prox = analise_proximidade_alvos(sol)
        self.assertIn("pilhas", prox)
        self.assertIn("estatisticas", prox)
        self.assertIn("criterio_adotado", prox)

        p1_info = prox["pilhas"]["P1"]
        self.assertEqual(p1_info["grupo"], "S1")
        self.assertEqual(p1_info["SiO2"]["classificacao"], "Excelente")
        self.assertEqual(p1_info["Al2O3"]["classificacao"], "Excelente")

        p6_info = prox["pilhas"]["P6"]
        self.assertEqual(p6_info["grupo"], "S2")
        self.assertEqual(p6_info["SiO2"]["classificacao"], "Excelente")
        self.assertEqual(p6_info["Al2O3"]["classificacao"], "Afastado")  # desvio 0.30 > 0.25


if __name__ == "__main__":
    unittest.main()
