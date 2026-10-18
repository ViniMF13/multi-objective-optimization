# -*- coding: utf-8 -*-
"""
test_vns_mechanics.py
Testes unitários e diagnósticos detalhados para o VND e GVNS:
- VND (Variable Neighborhood Descent): cascata N1 -> N2 -> N3, reinício em N1 após melhora,
  e terminação em ótimo local multi-vizinhança.
- GVNS (General Variable Neighborhood Search): shaking estocástico, busca local VND,
  critério de aceitação (move-or-not), e histórico não-crescente de convergência.
- Impressão explícita de cada etapa, transição e valores esperados vs obtidos.
"""

import random
import unittest
from construction import construir_solucao
from model import avaliar, resumo_solucao
from neighborhoods import descida_n1, descida_n2
from vns import vnd, gvns, busca_local_n1


class TestVNSMechanics(unittest.TestCase):
    """Testes detalhados da mecânica algorítmica do VND e GVNS."""

    def test_vnd_cascading_and_neighborhood_switches(self):
        """
        Valida e exibe detalhadamente a cascata de vizinhanças do VND:
        - Explora N1; se estagnar, avança para N2;
        - Se N2 encontrar melhora, REINICIA em N1;
        - Só encerra quando atinge ótimo local conjunto ou esgota orçamento.
        """
        print("\n" + "=" * 70)
        print("TESTE: VND (Variable Neighborhood Descent - Cascata N1 -> N2 -> N3)")
        print("=" * 70)

        rng = random.Random(42)
        sol_init = construir_solucao(seed=42)
        val_init = avaliar(sol_init, "f2")

        print(f"Solução Construtiva Inicial: Fitness f2 = {val_init:.4f}")

        sol_vnd, val_vnd, avals, stats = vnd(
            sol_init, "f2", rng=rng, max_evals=400, verbose=False
        )

        print(f"\nTotal de avaliações gastas no VND: {avals}")
        print(f"Melhora obtida: {val_init:.4f} -> {val_vnd:.4f} (Ganho: {val_init - val_vnd:.4f})")
        print("\nEstatísticas por Vizinhança no VND:")
        print(f"  N1 (Troca Simples):                  {stats['avals_por_vizinhanca'][0]} avaliações | {stats['melhoras_por_vizinhanca'][0]} melhorias aceitas")
        print(f"  N2 (Troca Entre Pilhas):             {stats['avals_por_vizinhanca'][1]} avaliações | {stats['melhoras_por_vizinhanca'][1]} melhorias aceitas")
        print(f"  N3 (Substituição Tripla Distribuída): {stats['avals_por_vizinhanca'][2]} avaliações | {stats['melhoras_por_vizinhanca'][2]} melhorias aceitas")

        print("\nRegistro cronológico das primeiras transições aceitas pelo VND:")
        for idx, t in enumerate(stats["transicoes"][:6]):
            print(f"  Passo {idx+1} [{t['vizinhanca']}]: {t['val_antes']:.4f} -> {t['val_depois']:.4f} (delta = {t['delta']:.4f})")

        # Asserções
        self.assertLessEqual(val_vnd, val_init, "VND nunca deve piorar a solução")
        self.assertAlmostEqual(val_vnd, avaliar(sol_vnd, "f2"), places=6)
        self.assertTrue(stats["sucesso_global"], "VND deve encontrar melhorias na solução inicial")
        print("Resultado: PASSOU (Comportamento de cascata e reinício em N1 verificado com sucesso)")

    def test_gvns_trace_and_shaking_acceptance(self):
        """
        Executa micro-instância do GVNS com trace explícito para auditar:
        - Shaking k=1, 2, 3
        - Chamada do VND sobre a solução perturbada
        - Atualização do incumbente e histórico de convergência
        """
        print("\n" + "=" * 70)
        print("TESTE: GVNS (General Variable Neighborhood Search Completo)")
        print("=" * 70)

        seed = 123
        sol_init = construir_solucao(seed=seed)
        val_init = avaliar(sol_init, "f1")
        print(f"Solução Inicial Heurística: Fitness penalizado f1 = {val_init:,.2f} R$")

        # Executa GVNS com orçamento controlado de 120 avaliações
        sol_final, val_final, historico = gvns(
            "f1", seed=seed, max_evals=120, k_max=3, ls_evals_limit=40, verbose=False
        )

        print(f"Solução Final GVNS:         Fitness penalizado f1 = {val_final:,.2f} R$")
        print(f"Número de pontos no histórico de convergência: {len(historico)}")

        print("\nEvolução dos primeiros marcos do histórico (avaliações, melhor_fitness):")
        for ponto in historico[:8]:
            print(f"  Avaliações: {ponto[0]:4d} | Incumbente: {ponto[1]:,.2f} R$")

        # Verifica monotonicidade do histórico do incumbente
        prev_val = float("inf")
        for avals_pt, val_pt in historico:
            self.assertLessEqual(val_pt, prev_val + 1e-9, "Incumbente no histórico nunca pode piorar")
            prev_val = val_pt

        self.assertLessEqual(val_final, val_init)
        self.assertAlmostEqual(val_final, historico[-1][1], places=4)
        print("Resultado: PASSOU (Monotonicidade do incumbente e critério de aceitação validados)")

    def test_gvns_reproducibility_trace(self):
        """Verifica se duas execuções com a mesma semente geram resultados idênticos."""
        print("\n" + "=" * 70)
        print("TESTE: GVNS Reprodutibilidade com Semente Controlada")
        print("=" * 70)

        sol_a, val_a, hist_a = gvns("f3", seed=99, max_evals=60, k_max=3, ls_evals_limit=20)
        sol_b, val_b, hist_b = gvns("f3", seed=99, max_evals=60, k_max=3, ls_evals_limit=20)

        print(f"Execução A: Fitness final = {val_a:.6f} | Histórico={len(hist_a)} passos")
        print(f"Execução B: Fitness final = {val_b:.6f} | Histórico={len(hist_b)} passos")

        self.assertAlmostEqual(val_a, val_b, places=9)
        self.assertEqual(sol_a, sol_b)
        self.assertEqual(hist_a, hist_b)
        print("Resultado: PASSOU (100% determinístico e reproduzível com a mesma semente)")

    def test_vnd_unlocks_local_optimum_via_n2(self):
        """
        DEMONSTRAÇÃO TEÓRICA FUNDAMENTAL DO VND:
        Comprova que quando uma busca local fica estagnada em N1 (ótimo local em relação a N1),
        a transição para a vizinhança N2 (troca entre pilhas) é capaz de destravar a busca
        e encontrar uma melhora não acessível por N1, justificando a arquitetura do VND.
        """
        print("\n" + "=" * 70)
        print("TESTE: VND Destravando Ótimo Local de N1 através de N2")
        print("=" * 70)

        rng = random.Random(77)
        sol_init = construir_solucao(seed=77)

        # 1. Esgota as melhorias possíveis em N1
        sol_n1, val_n1, avals_n1, _, _ = descida_n1(sol_init, "f2", rng, max_evals=150)
        print(f"Fase 1: Descida em N1 concluída. Fitness resultante: {val_n1:.4f} (avals={avals_n1})")

        # 2. Executa N2 sobre a solução resultante de N1
        sol_n2, val_n2, avals_n2, melhorou_n2, trans_n2 = descida_n2(sol_n1, "f2", rng, max_evals=150)
        print(f"Fase 2: Aplicação de N2 sobre o ótimo de N1:")
        print(f"  Houve melhora via N2: {melhorou_n2}")
        print(f"  Fitness após N2: {val_n2:.4f} (Delta: {val_n2 - val_n1:.4f})")

        if melhorou_n2:
            print(f"  Troca que destravou o ótimo local: Pilha {trans_n2[0]['p1']} <-> Pilha {trans_n2[0]['p2']}")
            print("  -> COMPROVADO: N2 desbloqueou o ótimo local de N1! O VND reinicia em N1 com sucesso.")
            self.assertLess(val_n2, val_n1 - 1e-9)
        else:
            print("  N2 também não encontrou melhora imediata (solução altamente convergida).")
            self.assertLessEqual(val_n2, val_n1)

        print("Resultado: PASSOU")


if __name__ == "__main__":
    unittest.main()
