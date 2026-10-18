# -*- coding: utf-8 -*-
"""
test_neighborhoods.py
Testes unitários e diagnósticos detalhados para os operadores de vizinhança:
- Operadores de Perturbação (Shaking): N1, N2, N3
- Operadores de Descida de Busca Local (Descent): descida_n1, descida_n2, descida_n3
- Impressão explícita de entradas, transições executadas e saídas esperadas vs obtidas
  para fins de validação e depuração.
"""

import random
import unittest
from data import PILES, PILE_IDS, ELIGIBLE, MINERALS
from construction import construir_solucao
from model import avaliar, violacao_disponibilidade, violacao_qualidade, resumo_solucao
from neighborhoods import (
    copy_solution,
    shaking_n1,
    shaking_n2,
    shaking_n3,
    shaking_k,
    descida_n1,
    descida_n2,
    descida_n3,
    NEIGHBORHOODS,
)


class TestNeighborhoods(unittest.TestCase):
    """Testa individualmente cada operador de perturbação e descida."""

    def setUp(self):
        self.sol_base = construir_solucao(seed=42)

    def test_shaking_n1_explicit_trace(self):
        """Valida e exibe detalhadamente a perturbação em N1 (Troca Simples)."""
        rng = random.Random(101)
        print("\n" + "=" * 70)
        print("TESTE: shaking_n1 (Perturbação Leve N1 - Troca Simples)")
        print("=" * 70)

        sol_perturbada = shaking_n1(self.sol_base, rng)

        diffs = []
        for p in PILE_IDS:
            for idx, (m_ant, m_novo) in enumerate(zip(self.sol_base[p], sol_perturbada[p])):
                if m_ant != m_novo:
                    diffs.append((p, idx, m_ant, m_novo, PILES[p]["grupo"]))

        print(f"Pilha afetada: {diffs[0][0]} (Grupo {diffs[0][4]})")
        print(f"Posição alterada: índice {diffs[0][1]} de {len(self.sol_base[diffs[0][0]])} caminhões")
        print(f"Transição: {diffs[0][2]} -> {diffs[0][3]}")
        print(f"Elegíveis do grupo: {ELIGIBLE[diffs[0][4]]}")

        # Asserções
        self.assertEqual(len(diffs), 1, "Shaking N1 deve alterar rigorosamente 1 posição")
        p, idx, m_ant, m_novo, grp = diffs[0]
        self.assertNotEqual(m_ant, m_novo)
        self.assertIn(m_novo, ELIGIBLE[grp])
        print("Resultado: PASSOU (1 posição alterada, elegibilidade estritamente preservada)")

    def test_shaking_n2_cross_pile_swap_invariant(self):
        """Valida e exibe a troca entre pilhas N2 e o invariante de conservação de estoque."""
        rng = random.Random(202)
        print("\n" + "=" * 70)
        print("TESTE: shaking_n2 (Perturbação N2 - Troca Entre Pilhas Distintas)")
        print("=" * 70)

        sol_perturbada = shaking_n2(self.sol_base, rng, max_tentativas=50)

        diffs = []
        for p in PILE_IDS:
            for idx, (m_ant, m_novo) in enumerate(zip(self.sol_base[p], sol_perturbada[p])):
                if m_ant != m_novo:
                    diffs.append((p, idx, m_ant, m_novo, PILES[p]["grupo"]))

        print(f"Posições alteradas detectadas: {len(diffs)}")
        self.assertEqual(len(diffs), 2, "Shaking N2 deve trocar exatamente 2 posições entre 2 pilhas")
        (p1, i1, m1_ant, m1_novo, grp1), (p2, i2, m2_ant, m2_novo, grp2) = diffs

        print(f"Pilha 1: {p1} (Grupo {grp1}), pos={i1}, troca {m1_ant} -> {m1_novo}")
        print(f"Pilha 2: {p2} (Grupo {grp2}), pos={i2}, troca {m2_ant} -> {m2_novo}")
        self.assertEqual(m1_ant, m2_novo)
        self.assertEqual(m2_ant, m1_novo)
        self.assertIn(m1_novo, ELIGIBLE[grp1])
        self.assertIn(m2_novo, ELIGIBLE[grp2])

        # Verificação explícita do invariante de conservação do estoque
        uso_antes = {m: 0 for m in MINERALS}
        uso_depois = {m: 0 for m in MINERALS}
        for p in PILE_IDS:
            for m in self.sol_base[p]:
                uso_antes[m] += 1
            for m in sol_perturbada[p]:
                uso_depois[m] += 1

        self.assertEqual(uso_antes, uso_depois)
        print("Invariante de estoque: 100% CONSERVADO (nenhuma variação nas contagens globais)")
        print("Resultado: PASSOU")

    def test_descida_n1_first_improvement_trace(self):
        """Testa e imprime o passo a passo da busca local sistemática em N1."""
        rng = random.Random(303)
        print("\n" + "=" * 70)
        print("TESTE: descida_n1 (Busca Local First-Improvement em N1)")
        print("=" * 70)

        val_antes = avaliar(self.sol_base, "f2")
        print(f"Solução Inicial: Fitness penalizado para f2 = {val_antes:.4f}")

        sol_fim, val_fim, avals, melhorou, transicoes = descida_n1(
            self.sol_base, "f2", rng, max_evals=200
        )

        print(f"Avaliações de função gastas: {avals}")
        print(f"Houve melhora: {melhorou}")
        print(f"Número de transições aceitas: {len(transicoes)}")
        for idx, t in enumerate(transicoes[:5]):
            print(f"  Transição {idx+1}: Pilha {t['pilha']}[{t['posicao']}] ({t['de']} -> {t['para']}) "
                  f"| Fitness: {t['val_antes']:.4f} -> {t['val_depois']:.4f} (delta = {t['delta']:.4f})")
        if len(transicoes) > 5:
            print(f"  ... e mais {len(transicoes) - 5} transições de melhora.")

        print(f"Fitness Final Obtido: {val_fim:.4f} (Melhora Total: {val_fim - val_antes:.4f})")
        self.assertLessEqual(val_fim, val_antes)
        if melhorou:
            self.assertLess(val_fim, val_antes - 1e-9)
        print("Resultado: PASSOU (Descida estritamente não-crescente comprovada)")

    def test_descida_n2_cross_pile_local_search_trace(self):
        """Testa e imprime o passo a passo da busca local em N2 (troca entre pilhas)."""
        rng = random.Random(404)
        print("\n" + "=" * 70)
        print("TESTE: descida_n2 (Busca Local First-Improvement em N2)")
        print("=" * 70)

        val_antes = avaliar(self.sol_base, "f3")
        print(f"Solução Inicial: Fitness penalizado para f3 = {val_antes:.4f}")

        sol_fim, val_fim, avals, melhorou, transicoes = descida_n2(
            self.sol_base, "f3", rng, max_evals=200
        )

        print(f"Avaliações de função gastas: {avals}")
        print(f"Houve melhora via N2: {melhorou}")
        print(f"Transições aceitas em N2: {len(transicoes)}")
        for idx, t in enumerate(transicoes[:3]):
            print(f"  Troca {idx+1}: {t['p1']}[{t['i1']}] ({t['m1']}) <-> {t['p2']}[{t['i2']}] ({t['m2']}) "
                  f"| Fitness: {t['val_antes']:.4f} -> {t['val_depois']:.4f} (delta = {t['delta']:.4f})")

        print(f"Fitness Final Obtido: {val_fim:.4f}")
        self.assertLessEqual(val_fim, val_antes)
        print("Resultado: PASSOU (Trocas compatíveis entre pilhas executadas com sucesso)")

    def test_shaking_n3_explicit_trace(self):
        """Valida e exibe a perturbação N3 (Substituição Tripla Distribuída em 3 Pilhas Distintas)."""
        rng = random.Random(505)
        print("\n" + "=" * 70)
        print("TESTE: shaking_n3 (Perturbação Forte N3 - Substituição Tripla Distribuída)")
        print("=" * 70)

        sol_perturbada = shaking_n3(self.sol_base, rng)

        diffs = []
        for p in PILE_IDS:
            for idx, (m_ant, m_novo) in enumerate(zip(self.sol_base[p], sol_perturbada[p])):
                if m_ant != m_novo:
                    diffs.append((p, idx, m_ant, m_novo, PILES[p]["grupo"]))

        print(f"Número total de posições alteradas (Distância de Hamming): {len(diffs)}")
        pilhas_alteradas = {d[0] for d in diffs}
        print(f"Pilhas distintas afetadas: {sorted(list(pilhas_alteradas))} (Total: {len(pilhas_alteradas)})")

        self.assertEqual(len(diffs), 3, "Shaking N3 deve alterar rigorosamente 3 posições")
        self.assertEqual(len(pilhas_alteradas), 3, "Shaking N3 deve afetar exatamente 3 pilhas distintas")

        for idx, (p, pos, m_ant, m_novo, grp) in enumerate(diffs):
            print(f"  Pilha {p} (Grupo {grp}) [pos {pos}]: {m_ant} -> {m_novo} (Elegível: {m_novo in ELIGIBLE[grp]})")
            self.assertNotEqual(m_ant, m_novo)
            self.assertIn(m_novo, ELIGIBLE[grp])

        print("Resultado: PASSOU (3 pilhas distintas afetadas, d_H = 3, elegibilidade 100% garantida)")

    def test_descida_n3_explicit_trace(self):
        """Testa e imprime o passo a passo da busca local em N3 guiada por Deb."""
        rng = random.Random(606)
        print("\n" + "=" * 70)
        print("TESTE: descida_n3 (Busca Local First-Improvement em N3)")
        print("=" * 70)

        val_antes = avaliar(self.sol_base, "f1")
        print(f"Solução Inicial: Fitness penalizado para f1 = {val_antes:.4f}")

        sol_fim, val_fim, avals, melhorou, transicoes = descida_n3(
            self.sol_base, "f1", rng, max_evals=200
        )

        print(f"Avaliações de função gastas: {avals}")
        print(f"Houve melhora via N3: {melhorou}")
        print(f"Transições aceitas em N3: {len(transicoes)}")
        for idx, t in enumerate(transicoes[:3]):
            print(f"  Passo {idx+1}: Pilhas {t['pilhas']} | Fitness: {t['val_antes']:.4f} -> {t['val_depois']:.4f} (delta = {t['delta']:.4f})")

        print(f"Fitness Final Obtido: {val_fim:.4f}")
        self.assertLessEqual(val_fim, val_antes)
        print("Resultado: PASSOU (Descida em N3 com regras de Deb validada com sucesso)")


if __name__ == "__main__":
    unittest.main()
