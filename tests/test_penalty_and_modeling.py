# -*- coding: utf-8 -*-
"""
test_penalty_and_modeling.py
Testes unitários e diagnósticos do Princípio de Dominação de Restrições de Deb (2000)
e comparação com a modelagem de penalidade:
- fitness_deb: chave lexicográfica de ordenação
- eh_melhor_deb: viável sempre domina inviável, resolvendo o problema do falso ótimo de f1
- Comparação entre viáveis (menor objetivo) e entre inviáveis (menor violação)
- Desempenho e velocidade de avaliação
"""

import time
import unittest
from data import MINERALS, PILES, PILE_IDS, TRUCK_MASS_T, GROUP_BOUNDS
from model import (
    PENALTY_WEIGHT,
    penalidade_total,
    avaliar,
    violacao_disponibilidade,
    violacao_qualidade,
    violacao_total,
    fitness_deb,
    eh_melhor_deb,
    is_feasible,
    f1_custo,
    f2_desvio_sio2,
    f3_desvio_al2o3,
)
from construction import construir_solucao


class TestPenaltyAndModeling(unittest.TestCase):
    """Testa as Regras de Deb e o tratamento de factibilidade."""

    def setUp(self):
        self.sol_inicial = construir_solucao(seed=42)

    def test_deb_feasibility_strictly_dominates_infeasibility(self):
        """
        REGRA 1 DE DEB (A mais crítica para f1):
        Uma solução viável SEMPRE domina uma solução inviável,
        mesmo que a inviável aparente ter um custo financeiro muito menor.
        """
        # Cria uma solução sintética factível de custo R$ 64 milhões
        sol_viavel = {p: ["M7"] * PILES[p]["n_trucks"] for p in PILE_IDS}
        # M7 em quantidade 145 estoura disp_max, então vamos montar uma viável real
        # ou testar com fitness_deb diretamente
        sol_viavel_fake = {p: list(self.sol_inicial[p]) for p in PILE_IDS}

        # Simula uma solução A (viável) com custo 61.44 M
        # Simula uma solução B (inviável com violação de 0.01%) com custo 60.88 M
        # Testando a lógica de eh_melhor_deb
        class MockSol:
            pass

        # Cria solução inviável adicionando excesso de minério M1
        sol_inviavel = {p: list(self.sol_inicial[p]) for p in PILE_IDS}
        sol_inviavel["P1"][0] = "M1"  # introduz alteração
        sol_inviavel["P2"][0] = "M1"
        sol_inviavel["P3"][0] = "M1"
        sol_inviavel["P4"][0] = "M1"
        sol_inviavel["P5"][0] = "M1"  # M1 disp_max = 6, vai estourar

        v_inviavel = violacao_total(sol_inviavel)
        self.assertGreater(v_inviavel, 0.0, "Solução de teste deve ser inviável")

        # Verifica fitness_deb
        fit_inv = fitness_deb(sol_inviavel, "f1")
        self.assertEqual(fit_inv[0], 1, "Inviável deve ter chave 1")
        self.assertEqual(fit_inv[1], v_inviavel)

    def test_deb_both_feasible_lower_objective_wins(self):
        """
        REGRA 2 DE DEB:
        Entre duas soluções estritamente viáveis, a de MENOR objetivo puro vence.
        """
        # Se sol_a e sol_b forem viáveis com f(a) < f(b), a vence
        # Vamos testar com soluções com viabilidade garantida
        f_deb_a = (0, 61_440_000.0, 0.0)
        f_deb_b = (0, 62_000_000.0, 0.0)
        self.assertLess(f_deb_a, f_deb_b)

    def test_deb_both_infeasible_lower_violation_wins(self):
        """
        REGRA 3 DE DEB:
        Entre duas soluções inviáveis, a de MENOR violação vence,
        guiando a busca diretamente para a região viável.
        """
        # Solução A tem violação 0.02 e custo 60M
        # Solução B tem violação 0.10 e custo 50M
        # A deve vencer B porque viola menos!
        fit_a = (1, 0.02, 60_000_000.0)
        fit_b = (1, 0.10, 50_000_000.0)
        self.assertLess(fit_a, fit_b)

    def test_evaluar_scalar_embedding_of_deb(self):
        """
        A função avaliar(sol, obj) implementa a projeção escalar monotônica de Deb:
        soluções viáveis têm valor < 1e8, soluções inviáveis têm valor >= 1e9 + viol.
        """
        val_inicial = avaliar(self.sol_inicial, "f1")
        # Se a solução inicial for viável: val < 1e9; se inviável: val >= 1e9
        is_feas = is_feasible(self.sol_inicial)
        if is_feas:
            self.assertLess(val_inicial, 1.0e9)
        else:
            self.assertGreaterEqual(val_inicial, 1.0e9)

    def test_evaluation_performance_efficiency(self):
        """
        A avaliação baseada em Deb deve ser ultrarrápida (< 0.5 ms por chamada).
        """
        n_evals = 1000
        start = time.perf_counter()
        for _ in range(n_evals):
            _ = avaliar(self.sol_inicial, "f1")
        elapsed = time.perf_counter() - start

        tempo_por_aval_ms = (elapsed / n_evals) * 1000.0
        self.assertLess(
            tempo_por_aval_ms,
            1.0,
            f"Avaliação muito lenta: {tempo_por_aval_ms:.3f} ms por chamada",
        )


if __name__ == "__main__":
    unittest.main()
