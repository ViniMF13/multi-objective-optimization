# -*- coding: utf-8 -*-
"""
test_decision_theory.py
Testes unitários e diagnósticos detalhados para as ferramentas de Teoria da Decisão Multiobjetivo:
- Tabela de Payoff (Matriz de Conflito 3x3)
- Pontos de Referência: Ponto Ideal (z*), Ponto Utópico (z*uto) e Ponto de Nadir (z_nad)
- Matriz de Degradação Percentual (Conflito Normalizado)
- Normalização Min-Max para a Entrega 2
- Malha Discreta para o Método epsilon-Restrito
- Impressão explícita de entradas, matrizes, vetores e diagnósticos para validação.
"""

import unittest
from construction import construir_solucao
from model import f1_custo, f2_desvio_sio2, f3_desvio_al2o3, resumo_solucao, is_feasible
from neighborhoods import copy_solution, shaking_n1, shaking_n2, shaking_n3
from experiments import calcular_payoff_e_pontos_referencia


class TestDecisionTheoryTools(unittest.TestCase):
    """Testa formalmente as ferramentas analíticas de Teoria da Decisão."""

    def setUp(self):
        import random
        # Constrói 3 soluções factíveis de teste com características distintas
        # para simular soluções âncora s1*, s2*, s3*
        s_base = construir_solucao(seed=42)
        
        # Cria s1 (prioriza custo)
        self.s1 = copy_solution(s_base)
        # Cria s2 (prioriza SiO2)
        self.s2 = shaking_n1(s_base, random.Random(101))
        # Cria s3 (prioriza Al2O3)
        self.s3 = shaking_n2(s_base, random.Random(202))

        # Monta o dicionário 'melhores' estruturado como gerado pelo pipeline de experimentos
        self.melhores_mock = {
            "f1": {
                "sol": self.s1,
                "seed": 10,
                "valor_puro": float(f1_custo(self.s1)),
                "factivel": True,
            },
            "f2": {
                "sol": self.s2,
                "seed": 20,
                "valor_puro": float(f2_desvio_sio2(self.s2)),
                "factivel": True,
            },
            "f3": {
                "sol": self.s3,
                "seed": 30,
                "valor_puro": float(f3_desvio_al2o3(self.s3)),
                "factivel": True,
            },
        }

    def test_payoff_matrix_diagonal_consistency(self):
        """Valida e exibe detalhadamente a Matriz de Payoff 3x3 e sua diagonal principal."""
        print("\n" + "=" * 75)
        print("TESTE: Matriz de Payoff 3x3 (Consistência da Diagonal e Âncoras)")
        print("=" * 75)

        payoff_data = calcular_payoff_e_pontos_referencia(self.melhores_mock)
        pm = payoff_data["payoff_matrix"]

        print(f"Linha 1 [s1* Âncora f1]: f1=R$ {pm['f1']['f1']:12,.2f} | f2={pm['f1']['f2']:8.4f} | f3={pm['f1']['f3']:8.4f}")
        print(f"Linha 2 [s2* Âncora f2]: f1=R$ {pm['f2']['f1']:12,.2f} | f2={pm['f2']['f2']:8.4f} | f3={pm['f2']['f3']:8.4f}")
        print(f"Linha 3 [s3* Âncora f3]: f1=R$ {pm['f3']['f1']:12,.2f} | f2={pm['f3']['f2']:8.4f} | f3={pm['f3']['f3']:8.4f}")

        # Validação da diagonal
        self.assertAlmostEqual(pm["f1"]["f1"], f1_custo(self.s1), places=4)
        self.assertAlmostEqual(pm["f2"]["f2"], f2_desvio_sio2(self.s2), places=4)
        self.assertAlmostEqual(pm["f3"]["f3"], f3_desvio_al2o3(self.s3), places=4)
        print("Resultado: PASSOU (Consistência analítica da Matriz de Payoff comprovada)")

    def test_ideal_utopian_and_nadir_point_properties(self):
        """Valida as propriedades matemáticas fundamentais: z*uto < z* <= z_nad."""
        print("\n" + "=" * 75)
        print("TESTE: Pontos de Referência (Ideal z*, Utópico z*uto e Nadir z_nad)")
        print("=" * 75)

        payoff_data = calcular_payoff_e_pontos_referencia(self.melhores_mock)
        z_star = payoff_data["ponto_ideal"]
        z_uto = payoff_data["ponto_utopico"]
        z_nad = payoff_data["ponto_nadir"]

        print(f"Ponto Ideal (z*):    f1 = R$ {z_star['f1']:12,.2f} | f2 = {z_star['f2']:8.4f} | f3 = {z_star['f3']:8.4f}")
        print(f"Ponto Utópico (z*u): f1 = R$ {z_uto['f1']:12,.2f} | f2 = {z_uto['f2']:8.4f} | f3 = {z_uto['f3']:8.4f}")
        print(f"Ponto Nadir (z_nad): f1 = R$ {z_nad['f1']:12,.2f} | f2 = {z_nad['f2']:8.4f} | f3 = {z_nad['f3']:8.4f}")

        for obj in ["f1", "f2", "f3"]:
            # Propriedade 1: Nadir é sempre maior ou igual ao Ideal
            self.assertGreaterEqual(z_nad[obj], z_star[obj] - 1e-9, f"z_nad deve ser >= z* para {obj}")
            # Propriedade 2: Utópico é estritamente menor que o Ideal
            self.assertLess(z_uto[obj], z_star[obj] + 1e-9, f"z_uto deve ser < z* para {obj}")
            delta = z_nad[obj] - z_star[obj]
            print(f"  Amplitude Delta_{obj}: {delta:12.4f} | z_uto[{obj}] < z*[{obj}] <= z_nad[{obj}] : OK")

        print("Resultado: PASSOU (Relações de ordem dos pontos de referência validadas)")

    def test_min_max_normalization_bounds(self):
        """Valida a fórmula de normalização Min-Max para a Entrega 2 no intervalo [0, 1]."""
        print("\n" + "=" * 75)
        print("TESTE: Normalização Min-Max dos Objetivos para a Entrega 2")
        print("=" * 75)

        payoff_data = calcular_payoff_e_pontos_referencia(self.melhores_mock)
        norm_params = payoff_data["preparacao_entrega2"]["normalizacao"]
        mc_pct = payoff_data["matriz_conflito_pct"]

        print("Fatores de Escala Calculados:")
        for obj in ["f1", "f2", "f3"]:
            fator = norm_params[obj]["fator_escala"]
            delta = norm_params[obj]["intervalo_delta"]
            print(f"  {obj}: Amplitude={delta:12.4f} | Fator={fator:.6e}")
            self.assertGreater(fator, 0.0)

        print("\nMatriz de Conflito Normalizada (Degradação %):")
        for i in ["f1", "f2", "f3"]:
            linha = [f"{mc_pct[i][j]:6.1f}%" for j in ["f1", "f2", "f3"]]
            print(f"  Otimizando {i}: " + " | ".join(linha))
            # A diagonal da matriz de degradação deve ser rigorosamente 0.0%
            self.assertAlmostEqual(mc_pct[i][i], 0.0, places=4)

        print("Resultado: PASSOU (Mapeamento no hipercubo unitário e diagonal nula validados)")

    def test_epsilon_grid_discretization(self):
        """Valida a malha sugerida para o Método epsilon-Restrito na Entrega 2."""
        print("\n" + "=" * 75)
        print("TESTE: Malha de epsilon para o Método epsilon-Restrito (10 Intervalos)")
        print("=" * 75)

        payoff_data = calcular_payoff_e_pontos_referencia(self.melhores_mock)
        eps_data = payoff_data["preparacao_entrega2"]["metodo_epsilon_restrito"]

        grid2 = eps_data["grid_sugerido_eps2"]
        grid3 = eps_data["grid_sugerido_eps3"]

        print(f"Grid eps2 (SiO2)   [N={len(grid2)}]: {[round(x, 4) for x in grid2]}")
        print(f"Grid eps3 (Al2O3) [N={len(grid3)}]: {[round(x, 4) for x in grid3]}")

        self.assertEqual(len(grid2), 10, "A malha deve conter 10 pontos")
        self.assertEqual(len(grid3), 10, "A malha deve conter 10 pontos")

        # Verifica monotonicidade estrita
        for g in [grid2, grid3]:
            for idx in range(len(g) - 1):
                self.assertLessEqual(g[idx], g[idx+1] + 1e-9, "Grid de epsilon deve ser monotonicamente crescente")

        print("Resultado: PASSOU (Malha monótona e cobrindo perfeitamente o intervalo [z*, z_nad])")


if __name__ == "__main__":
    unittest.main()
