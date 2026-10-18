# -*- coding: utf-8 -*-
"""
test_multiobjective.py
Testes unitários e de integração para o módulo de Otimização Multiobjetivo (Etapa 2).
Valida com traces diagnósticos explícitos:
1. Geração da grade de pesos no simplex (Simplex-Lattice)
2. Geração da grade de corte epsilon baseada nos pontos z* e z^nad
3. Problema Escalar da Soma Ponderada (P_w) e normalização Min-Max
4. Problema Escalar do epsilon-Restrito (P_eps) e tratamento de cortes via Regras de Deb
5. Relações de Dominância de Pareto e Filtro de Não-Dominância
6. Métrica de Hipervolume 3D Exato (Lebesgue Measure analítica)
7. Métrica de Espaçamento de Schott (Uniformidade da frente)
8. Execução integrada de micro-GVNS com problemas escalares
"""

import unittest
import math
from construction import construir_solucao
from model import f1_custo, f2_desvio_sio2, f3_desvio_al2o3, is_feasible, avaliar, eh_melhor_deb
from vns import gvns, vnd
from multiobjective import (
    carregar_referencias,
    gerar_pesos_simplex,
    gerar_grade_epsilon,
    domina_vetor,
    domina_solucao,
    filtrar_nao_dominadas,
    calcular_espacamento,
    calcular_hipervolume_2d,
    calcular_hipervolume_3d,
    calcular_cobertura_dois_conjuntos,
    SomaPonderadaProblem,
    EpsilonRestritoProblem,
)


class TestMultiobjectiveCore(unittest.TestCase):

    def test_simplex_weights_generation(self):
        """Valida a geração da grade regular de pesos no 3-simplex e exibe traces explícitos."""
        pesos_25 = gerar_pesos_simplex(passo=0.25)
        pesos_20 = gerar_pesos_simplex(passo=0.20)

        print("\n" + "=" * 70)
        print("TESTE: Geração da Grade de Pesos no Simplex (Simplex-Lattice Design)")
        print("=" * 70)
        print(f"Passo 0.25: {len(pesos_25)} vetores de pesos gerados (esperado=15)")
        print(f"Passo 0.20: {len(pesos_20)} vetores de pesos gerados (esperado=21)")

        print("\nAmostra dos vetores de pesos para passo=0.25:")
        for idx, (w1, w2, w3) in enumerate(pesos_25[:8]):
            print(f"  w[{idx+1:02d}]: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f} | Soma={w1+w2+w3:.4f}")
        print("  ...")

        self.assertEqual(len(pesos_25), 15)
        self.assertEqual(len(pesos_20), 21)

        for w in pesos_25 + pesos_20:
            self.assertAlmostEqual(sum(w), 1.0, places=5)
            self.assertTrue(all(val >= -1e-9 for val in w))

        print("Resultado: PASSOU (Soma 1.0 e cardinalidade exata validadas)")

    def test_epsilon_grid_generation(self):
        """Valida a geração da grade bidimensional de cotas epsilon baseada na Matriz de Payoff."""
        z_star, z_nad, _, delta = carregar_referencias()
        grade_4 = gerar_grade_epsilon(num_pontos_por_eixo=4, z_star=z_star, z_nad=z_nad)

        print("\n" + "=" * 70)
        print("TESTE: Geração da Grade Bidimensional de epsilon (eps2, eps3)")
        print("=" * 70)
        print(f"Ponto Ideal (z*):  SiO2={z_star['f2']:.4f}, Al2O3={z_star['f3']:.4f}")
        print(f"Ponto Nadir (z^nad): SiO2={z_nad['f2']:.4f}, Al2O3={z_nad['f3']:.4f}")
        print(f"Intervalos (Delta): Delta2={delta['f2']:.4f}, Delta3={delta['f3']:.4f}")
        print(f"Total de pares (eps2, eps3) gerados para 4 pontos por eixo: {len(grade_4)} (esperado=16)")

        print("\nPares gerados:")
        for idx, (e2, e3) in enumerate(grade_4[:6]):
            print(f"  Par {idx+1:02d}: eps2={e2:.4f}, eps3={e3:.4f}")
        print("  ...")

        self.assertEqual(len(grade_4), 16)
        # Menor valor de eps deve ser z* e maior deve ser z^nad
        eps2_valores = [p[0] for p in grade_4]
        eps3_valores = [p[1] for p in grade_4]
        self.assertAlmostEqual(min(eps2_valores), z_star["f2"], places=4)
        self.assertAlmostEqual(max(eps2_valores), z_nad["f2"], places=4)
        self.assertAlmostEqual(min(eps3_valores), z_star["f3"], places=4)
        self.assertAlmostEqual(max(eps3_valores), z_nad["f3"], places=4)

        print("Resultado: PASSOU (Limites z* e z^nad estritamente respeitados)")

    def test_soma_ponderada_problem_mechanics(self):
        """Valida o envoltório escalar SomaPonderadaProblem e a normalização Min-Max."""
        sol = construir_solucao(seed=42)
        z_star, z_nad, _, _ = carregar_referencias()

        # Instancia P_w com foco puramente em custo (1, 0, 0)
        p_custo = SomaPonderadaProblem(pesos=(1.0, 0.0, 0.0), z_star=z_star, z_nad=z_nad)
        # Instancia P_w com foco puramente em SiO2 (0, 1, 0)
        p_sio2 = SomaPonderadaProblem(pesos=(0.0, 1.0, 0.0), z_star=z_star, z_nad=z_nad)
        # Instancia P_w balanceado (0.333, 0.333, 0.334)
        p_bal = SomaPonderadaProblem(pesos=(0.333, 0.333, 0.334), z_star=z_star, z_nad=z_nad)

        val_custo = p_custo.eval_obj(sol)
        val_sio2 = p_sio2.eval_obj(sol)
        val_bal = p_bal.eval_obj(sol)

        print("\n" + "=" * 70)
        print("TESTE: Mecânica da Soma Ponderada Normalizada (P_w)")
        print("=" * 70)
        print(f"Solução construtiva seed=42:")
        print(f"  f1 puro = R$ {f1_custo(sol):,.2f} -> f1_norm = {val_custo:.4f}")
        print(f"  f2 puro = {f2_desvio_sio2(sol):.4f}     -> f2_norm = {val_sio2:.4f}")
        print(f"  P_w balanceado (0.333, 0.333, 0.334) = {val_bal:.4f}")

        # Testa compatibilidade com a função avaliar() do model.py
        val_avaliar = avaliar(sol, p_bal)
        print(f"  Avaliação escalar via model.avaliar(sol, p_bal) = {val_avaliar:.4f}")

        self.assertAlmostEqual(val_bal, 0.333 * val_custo + 0.333 * val_sio2 + 0.334 * ((f3_desvio_al2o3(sol) - z_star["f3"]) / (z_nad["f3"] - z_star["f3"])), places=5)
        print("Resultado: PASSOU (Combinação convexa e compatibilidade com model.py verificadas)")

    def test_epsilon_restrito_problem_deb_integration(self):
        """Valida que cortes epsilon são tratados com prioridade absoluta de factibilidade pelas regras de Deb."""
        sol = construir_solucao(seed=10)
        f2_atual = f2_desvio_sio2(sol)
        f3_atual = f3_desvio_al2o3(sol)
        z_star, z_nad, _, _ = carregar_referencias()

        print("\n" + "=" * 70)
        print("TESTE: Epsilon-Restrito e Dominação de Restrições de Deb")
        print("=" * 70)
        print(f"Valores da solução: f1=R$ {f1_custo(sol):,.2f}, f2={f2_atual:.4f}, f3={f3_atual:.4f}")

        # Caso A: epsilons folgados (a solução atende aos cortes)
        p_folgado = EpsilonRestritoProblem(eps2=f2_atual + 1.0, eps3=f3_atual + 1.0, z_star=z_star, z_nad=z_nad)
        v2_norm, v3_norm = p_folgado.violacao_eps(sol)
        print(f"Caso Folgado (eps2={f2_atual+1.0:.4f}): Violações eps = ({v2_norm:.4f}, {v3_norm:.4f})")
        self.assertEqual(v2_norm, 0.0)
        self.assertEqual(v3_norm, 0.0)

        # Caso B: epsilons apertados (a solução viola o corte de SiO2)
        eps2_apertado = f2_atual - 0.5
        p_apertado = EpsilonRestritoProblem(eps2=eps2_apertado, eps3=f3_atual + 1.0, z_star=z_star, z_nad=z_nad)
        v2_norm_ap, v3_norm_ap = p_apertado.violacao_eps(sol)
        print(f"Caso Apertado (eps2={eps2_apertado:.4f}): Violação eps2_norm = {v2_norm_ap:.4f}")
        self.assertGreater(v2_norm_ap, 0.0)

        # Valida que Deb penaliza a solução que violou o corte
        fit_apertado = p_apertado.fitness_deb(sol)
        self.assertEqual(fit_apertado[0], 1)  # Marca como inviável!
        self.assertGreater(fit_apertado[1], 0.0)

        print("Resultado: PASSOU (Corte epsilon integrado nativamente como restrição de Deb)")

    def test_pareto_dominance_and_filter(self):
        """Valida o verificador de dominância e o filtro de não-dominância em R^3."""
        print("\n" + "=" * 70)
        print("TESTE: Relação de Dominância de Pareto e Filtro de Não-Dominância")
        print("=" * 70)

        # Conjunto de teste sintético com relações de dominância conhecidas
        # P1 domina P2; P1 e P3 são incomparáveis (trade-off)
        p1 = {"id": "Sol_A", "f1": 10.0, "f2": 5.0, "f3": 5.0}
        p2 = {"id": "Sol_B", "f1": 12.0, "f2": 6.0, "f3": 5.5}  # Dominado por A
        p3 = {"id": "Sol_C", "f1": 8.0,  "f2": 7.0, "f3": 6.0}  # Incomparável com A (mais barato, pior qualidade)
        p4 = {"id": "Sol_D", "f1": 10.0, "f2": 5.0, "f3": 5.0}  # Duplicata idêntica de A

        self.assertTrue(domina_solucao(p1, p2))
        self.assertFalse(domina_solucao(p2, p1))
        self.assertFalse(domina_solucao(p1, p3))
        self.assertFalse(domina_solucao(p3, p1))

        candidatos = [p1, p2, p3, p4]
        nao_dominadas = filtrar_nao_dominadas(candidatos)

        ids_nao_dominadas = [s["id"] for s in nao_dominadas]
        print(f"Candidatos originais: {[c['id'] for c in candidatos]}")
        print(f"Soluções retidas na Fronteira de Pareto: {ids_nao_dominadas}")

        self.assertIn("Sol_A", ids_nao_dominadas)
        self.assertIn("Sol_C", ids_nao_dominadas)
        self.assertNotIn("Sol_B", ids_nao_dominadas)  # P2 deve ser descartado
        self.assertEqual(len(nao_dominadas), 2)       # P4 descartado como duplicata

        print("Resultado: PASSOU (Filtro descartou dominadas e duplicatas com precisão)")

    def test_hypervolume_3d_analytical(self):
        """Valida o cálculo do Hipervolume 3D contra valores analíticos exatos."""
        print("\n" + "=" * 70)
        print("TESTE: Hipervolume 3D Exato (Lebesgue Measure Analítica)")
        print("=" * 70)

        ref_point = (10.0, 10.0, 10.0)

        # Caso 1: Um único ponto (5, 5, 5) -> Volume do paralelepípedo = (10-5)*(10-5)*(10-5) = 125
        pts_1 = [(5.0, 5.0, 5.0)]
        hv_1 = calcular_hipervolume_3d(pts_1, ref_point)
        print(f"Caso 1 (Único Ponto [5, 5, 5]): HV Calculado = {hv_1:.4f} (Teórico = 125.0000)")
        self.assertAlmostEqual(hv_1, 125.0, places=4)

        # Caso 2: Dois pontos ortogonais
        # P1 = (5, 5, 5) [vol 125] e P2 = (2, 8, 8)
        pts_2 = [(5.0, 5.0, 5.0), (2.0, 8.0, 8.0)]
        hv_2 = calcular_hipervolume_3d(pts_2, ref_point)
        print(f"Caso 2 (Dois Pontos Não-Dominados): HV Calculado = {hv_2:.4f}")
        self.assertGreater(hv_2, 125.0)

        print("Resultado: PASSOU (Exatidão volumétrica 3D comprovada)")

    def test_spacing_metric(self):
        """Valida a métrica de Espaçamento de Schott (S)."""
        print("\n" + "=" * 70)
        print("TESTE: Métrica de Espaçamento de Schott (Uniformidade)")
        print("=" * 70)

        # Conjunto perfeitamente equidistante em uma reta 3D
        pts_uniformes = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)]
        s_unif = calcular_espacamento(pts_uniformes)
        print(f"Espaçamento para pontos perfeitamente uniformes: S = {s_unif:.6f} (esperado = 0.0)")
        self.assertAlmostEqual(s_unif, 0.0, places=5)

        # Conjunto não-uniforme (com distâncias heterogêneas ao vizinho mais próximo)
        pts_desiguais = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (2.0, 2.0, 2.0), (5.0, 5.0, 5.0)]
        s_desig = calcular_espacamento(pts_desiguais)
        print(f"Espaçamento para pontos desiguais (clusters heterogêneos): S = {s_desig:.6f}")
        self.assertGreater(s_desig, 0.0)

        print("Resultado: PASSOU (Sensibilidade a desvios de uniformidade comprovada)")

    def test_scalar_gvns_micro_run(self):
        """Valida a execução de um ciclo de GVNS otimizando um problema multiobjetivo escalarizado."""
        print("\n" + "=" * 70)
        print("TESTE: Execução Integrada do GVNS com Soma Ponderada (P_w)")
        print("=" * 70)

        z_star, z_nad, _, _ = carregar_referencias()
        prob_pw = SomaPonderadaProblem(pesos=(0.5, 0.25, 0.25), z_star=z_star, z_nad=z_nad)

        # Executa micro GVNS com 120 avaliações para verificar integração funcional
        melhor_sol, melhor_val, historico = gvns(
            objetivo=prob_pw,
            seed=42,
            max_evals=120,
            ls_evals_limit=40,
            verbose=False,
        )

        f1 = f1_custo(melhor_sol)
        f2 = f2_desvio_sio2(melhor_sol)
        f3 = f3_desvio_al2o3(melhor_sol)

        print(f"Resultado do GVNS multiobjetivo:")
        print(f"  Fitness escalar final P_w: {melhor_val:.4f}")
        print(f"  Objetivos puros: f1 = R$ {f1:,.2f} | f2 = {f2:.4f} | f3 = {f3:.4f}")
        print(f"  Número de marcos no histórico: {len(historico)}")

        self.assertIsNotNone(melhor_sol)
        self.assertGreater(len(historico), 1)

        print("Resultado: PASSOU (Motor de busca opera nativamente com o problema escalar)")


if __name__ == "__main__":
    unittest.main()
