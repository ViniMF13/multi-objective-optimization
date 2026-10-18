# -*- coding: utf-8 -*-
"""
multiobjective.py
Módulo de Otimização Multiobjetivo para a Etapa 2:
1. Métodos de Escalarização:
   - Soma Ponderada (Weighted Sum - P_w) com Normalização Min-Max
   - Método epsilon-Restrito (epsilon-Constraint - P_eps) com cortes de qualidade integrados a Deb
2. Geradores de Parâmetros:
   - Simplex de Pesos (Lattice Design na 3-simplex)
   - Grade Bidimensional de epsilon (amostragem em [z*, z^nad])
3. Ferramentas Analíticas e Teoria de Pareto:
   - Verificador de Dominância de Pareto (estrita, fraca e incomparabilidade)
   - Filtro de Não-Dominância (Fast Non-Dominated Sorting)
   - Cálculo Exato do Hipervolume 3D (Lebesgue Measure via fatiamento)
   - Métrica de Espaçamento de Schott (Uniformidade)
   - Métrica de Cobertura C(A, B) de Zitzler
"""

import json
import math
import os
from data import PILE_IDS, GROUP_BOUNDS
from model import (
    f1_custo,
    f2_desvio_sio2,
    f3_desvio_al2o3,
    violacao_total,
    violacao_disponibilidade,
    violacao_qualidade,
    is_feasible as is_feasible_fisico,
)

# Valores de referência padrão (obtidos rigorosamente na Etapa 1)
DEFAULT_Z_STAR = {"f1": 60840000.0, "f2": 2.4405207093717096, "f3": 1.472806603118444}
DEFAULT_Z_NAD = {"f1": 65600000.0, "f2": 6.49726929176314, "f3": 5.410476114109795}
DEFAULT_Z_UTO = {"f1": 60792400.0, "f2": 2.3999532235477954, "f3": 1.4334299080085304}


def carregar_referencias(caminho_json="outputs/resultados_entrega1_v2.json"):
    """
    Carrega os pontos Ideal (z*), Nadir (z^nad) e Utópico (z^uto) gerados na Etapa 1.
    Retorna (z_star, z_nad, z_uto, delta).
    """
    if os.path.exists(caminho_json):
        try:
            with open(caminho_json, "r", encoding="utf-8") as f:
                dados = json.load(f)
            ref = dados.get("payoff_e_pontos_referencia", {})
            z_star = ref.get("ponto_ideal", DEFAULT_Z_STAR)
            z_nad = ref.get("ponto_nadir", DEFAULT_Z_NAD)
            z_uto = ref.get("ponto_utopico", DEFAULT_Z_UTO)
        except Exception:
            z_star, z_nad, z_uto = DEFAULT_Z_STAR, DEFAULT_Z_NAD, DEFAULT_Z_UTO
    else:
        z_star, z_nad, z_uto = DEFAULT_Z_STAR, DEFAULT_Z_NAD, DEFAULT_Z_UTO

    delta = {
        k: (z_nad[k] - z_star[k]) if (z_nad[k] - z_star[k]) > 1e-9 else 1.0
        for k in ["f1", "f2", "f3"]
    }
    return z_star, z_nad, z_uto, delta


# ---------------------------------------------------------------------------
# 1. Classes Base e Problemas Escalares
# ---------------------------------------------------------------------------

class ScalarProblem:
    """
    Classe base abstrata para problemas escalares no framework VNS/GVNS/Deb.
    Fornece interface padrão com suporte a polimorfismo para fitness_deb e avaliar.
    """
    def __init__(self, nome="ScalarProblem"):
        self.nome = nome

    def eval_obj(self, sol):
        """Retorna o valor escalar da função objetivo agregada (mono-objetivo)."""
        raise NotImplementedError

    def violacao(self, sol):
        """Retorna a violação total de restrições (físicas e cortes adicionais)."""
        return violacao_total(sol)

    def is_feasible(self, sol, tol=1e-9):
        """Retorna True se todas as restrições (físicas + cortes) forem respeitadas."""
        return self.violacao(sol) <= tol

    def fitness_deb(self, sol):
        """
        Gera a chave lexicográfica canônica de Deb (2000):
        - Se viável:   (0, eval_obj(sol), 0.0)
        - Se inviável: (1, violacao(sol), eval_obj(sol))
        """
        f = self.eval_obj(sol)
        v = self.violacao(sol)
        if v <= 1e-9:
            return (0, float(f), 0.0)
        else:
            return (1, float(v), float(f))

    def eh_melhor_deb(self, sol_a, sol_b, tol=1e-9):
        """Critério de dominação de restrições de Deb."""
        fit_a = self.fitness_deb(sol_a)
        fit_b = self.fitness_deb(sol_b)

        # 1. Factibilidade tem prioridade absoluta
        if fit_a[0] < fit_b[0]:
            return True
        if fit_a[0] > fit_b[0]:
            return False

        # 2. Ambas viáveis: menor objetivo vence
        if fit_a[0] == 0:
            return fit_a[1] < fit_b[1] - tol

        # 3. Ambas inviáveis: menor violação agregada vence
        if fit_a[1] < fit_b[1] - tol:
            return True
        elif fit_a[1] > fit_b[1] + tol:
            return False
        else:
            return fit_a[2] < fit_b[2] - tol

    def avaliar(self, sol):
        """Projeção escalar monotônica de Deb para VND/GVNS."""
        f = self.eval_obj(sol)
        v = self.violacao(sol)
        if v <= 1e-9:
            return float(f)
        else:
            return 1.0e9 + float(v)

    def __call__(self, sol):
        return self.eval_obj(sol)


class SomaPonderadaProblem(ScalarProblem):
    """
    Abordagem da Soma Ponderada (P_w) com Normalização Min-Max:
    min P_w(s; lambda) = lambda_1 * f1^norm(s) + lambda_2 * f2^norm(s) + lambda_3 * f3^norm(s)
    sujeito às restrições físicas do sistema.
    """
    def __init__(self, pesos, z_star=None, z_nad=None, nome=None):
        if z_star is None or z_nad is None:
            zs, zn, _, d = carregar_referencias()
            self.z_star = zs
            self.z_nad = zn
            self.delta = d
        else:
            self.z_star = dict(z_star)
            self.z_nad = dict(z_nad)
            self.delta = {
                k: (self.z_nad[k] - self.z_star[k]) if (self.z_nad[k] - self.z_star[k]) > 1e-9 else 1.0
                for k in ["f1", "f2", "f3"]
            }

        self.pesos = tuple(float(w) for w in pesos)
        if nome is None:
            nome = f"Pw_w=({self.pesos[0]:.2f},{self.pesos[1]:.2f},{self.pesos[2]:.2f})"
        super().__init__(nome=nome)

    def eval_obj(self, sol):
        f1 = f1_custo(sol)
        f2 = f2_desvio_sio2(sol)
        f3 = f3_desvio_al2o3(sol)

        f1_norm = (f1 - self.z_star["f1"]) / self.delta["f1"]
        f2_norm = (f2 - self.z_star["f2"]) / self.delta["f2"]
        f3_norm = (f3 - self.z_star["f3"]) / self.delta["f3"]

        return (
            self.pesos[0] * f1_norm
            + self.pesos[1] * f2_norm
            + self.pesos[2] * f3_norm
        )

    def violacao(self, sol):
        return violacao_total(sol)


class EpsilonRestritoProblem(ScalarProblem):
    """
    Abordagem do epsilon-Restrito (P_eps):
    min f1^norm(s) = (f1(s) - z1*) / Delta1
    sujeito a:
      f2(s) <= eps2
      f3(s) <= eps3
      s in S (restrições físicas de massa, elegibilidade, qualidade e estoque).

    As violações de epsilon entram diretamente na violação agregada de Deb,
    normalizadas por Delta_k para manter a comensurabilidade.
    """
    def __init__(self, eps2, eps3, z_star=None, z_nad=None, nome=None):
        if z_star is None or z_nad is None:
            zs, zn, _, d = carregar_referencias()
            self.z_star = zs
            self.z_nad = zn
            self.delta = d
        else:
            self.z_star = dict(z_star)
            self.z_nad = dict(z_nad)
            self.delta = {
                k: (self.z_nad[k] - self.z_star[k]) if (self.z_nad[k] - self.z_star[k]) > 1e-9 else 1.0
                for k in ["f1", "f2", "f3"]
            }

        self.eps2 = float(eps2)
        self.eps3 = float(eps3)
        if nome is None:
            nome = f"Peps_eps2={self.eps2:.4f}_eps3={self.eps3:.4f}"
        super().__init__(nome=nome)

    def eval_obj(self, sol):
        f1 = f1_custo(sol)
        return (f1 - self.z_star["f1"]) / self.delta["f1"]

    def violacao_eps(self, sol):
        """Retorna tupla (v2_norm, v3_norm) das violações das restrições de corte epsilon."""
        f2 = f2_desvio_sio2(sol)
        f3 = f3_desvio_al2o3(sol)
        v2 = max(0.0, f2 - self.eps2)
        v3 = max(0.0, f3 - self.eps3)
        return (v2 / self.delta["f2"], v3 / self.delta["f3"])

    def violacao(self, sol):
        """Violação total = violações físicas + violações de epsilon normalizadas."""
        v_fisica = violacao_total(sol)
        v2_norm, v3_norm = self.violacao_eps(sol)
        return v_fisica + v2_norm + v3_norm

    def is_feasible(self, sol, tol=1e-9):
        if violacao_total(sol) > tol:
            return False
        f2 = f2_desvio_sio2(sol)
        f3 = f3_desvio_al2o3(sol)
        return (f2 <= self.eps2 + tol) and (f3 <= self.eps3 + tol)


# ---------------------------------------------------------------------------
# 2. Geradores de Parâmetros e Grades
# ---------------------------------------------------------------------------

def gerar_pesos_simplex(passo=0.25):
    """
    Gera uma grade regular (Simplex-Lattice Design) de vetores de peso (w1, w2, w3)
    tais que w1 + w2 + w3 = 1.0 e w_i >= 0.

    Exemplos:
    - passo = 0.25: 15 vetores de pesos (vertices, arestas e centroides)
    - passo = 0.20: 21 vetores de pesos
    """
    n_passos = int(round(1.0 / passo))
    pesos = []
    for i in range(n_passos + 1):
        for j in range(n_passos + 1 - i):
            k = n_passos - i - j
            w1 = round(i * passo, 4)
            w2 = round(j * passo, 4)
            w3 = round(k * passo, 4)
            # Normalização estrita para garantir soma exata de 1.0
            total = w1 + w2 + w3
            if abs(total - 1.0) < 1e-6:
                pesos.append((w1, w2, w3))
    return pesos


def gerar_grade_epsilon(num_pontos_por_eixo=5, z_star=None, z_nad=None):
    """
    Gera grade bidimensional de cotas (eps2, eps3) uniformemente espaçadas no intervalo
    [z*_k, z^nad_k] para k in {2, 3}.
    Para num_pontos_por_eixo=4: 16 pares (eps2, eps3).
    Para num_pontos_por_eixo=5: 25 pares (eps2, eps3).
    """
    if z_star is None or z_nad is None:
        zs, zn, _, _ = carregar_referencias()
    else:
        zs, zn = z_star, z_nad

    grid_eps2 = [
        zs["f2"] + (i / (num_pontos_por_eixo - 1)) * (zn["f2"] - zs["f2"])
        for i in range(num_pontos_por_eixo)
    ]
    grid_eps3 = [
        zs["f3"] + (j / (num_pontos_por_eixo - 1)) * (zn["f3"] - zs["f3"])
        for j in range(num_pontos_por_eixo)
    ]

    pares = []
    for e2 in grid_eps2:
        for e3 in grid_eps3:
            pares.append((round(e2, 6), round(e3, 6)))
    return pares


# ---------------------------------------------------------------------------
# 3. Dominância de Pareto e Filtro de Não-Dominância
# ---------------------------------------------------------------------------

def domina_vetor(u, v, tol=1e-9):
    """
    Retorna True se o vetor u dominar estritamente o vetor v segundo Pareto:
    u_i <= v_i + tol para todo i, e existe ao menos um j tal que u_j < v_j - tol.
    """
    menor_ou_igual = all(u[i] <= v[i] + tol for i in range(len(u)))
    estritamente_menor = any(u[i] < v[i] - tol for i in range(len(u)))
    return menor_ou_igual and estritamente_menor


def domina_solucao(sol_a, sol_b, tol=1e-9):
    """
    Verifica se a solução a domina a solução b no espaço de objetivos (f1, f2, f3).
    Aceita dicionários de solução, resumos com chaves 'f1','f2','f3' ou tuplas (f1, f2, f3).
    """
    def extrair_vetor(s):
        if isinstance(s, (tuple, list)):
            return s[:3]
        if isinstance(s, dict) and "f1" in s and "f2" in s and "f3" in s:
            return (s["f1"], s["f2"], s["f3"])
        return (f1_custo(s), f2_desvio_sio2(s), f3_desvio_al2o3(s))

    va = extrair_vetor(sol_a)
    vb = extrair_vetor(sol_b)
    return domina_vetor(va, vb, tol=tol)


def filtrar_nao_dominadas(candidatos, tol=1e-9):
    """
    Filtro de Não-Dominância (Pareto Filter):
    Recebe uma lista de candidatos (dicionários contendo 'f1', 'f2', 'f3' e metadados)
    e retorna apenas o conjunto não-dominado (Fronteira de Pareto aproximada).
    Descarta duplicatas numéricas com distância de Chebyshev <= tol.
    """
    if not candidatos:
        return []

    # Extrai vetor de objetivos para cada candidato
    pontos = []
    for c in candidatos:
        if isinstance(c, dict) and "f1" in c:
            v = (float(c["f1"]), float(c["f2"]), float(c["f3"]))
        elif isinstance(c, (tuple, list)):
            v = (float(c[0]), float(c[1]), float(c[2]))
        else:
            v = (float(f1_custo(c)), float(f2_desvio_sio2(c)), float(f3_desvio_al2o3(c)))
        pontos.append(v)

    n = len(candidatos)
    eh_dominado = [False] * n

    for i in range(n):
        if eh_dominado[i]:
            continue
        pi = pontos[i]
        for j in range(n):
            if i == j or eh_dominado[j]:
                continue
            pj = pontos[j]
            # Se pi dominar pj, descarta pj
            if domina_vetor(pi, pj, tol=tol):
                eh_dominado[j] = True
            # Se forem idênticos (duplicata), descarta o de índice maior
            elif (
                abs(pi[0] - pj[0]) <= tol
                and abs(pi[1] - pj[1]) <= tol
                and abs(pi[2] - pj[2]) <= tol
                and j > i
            ):
                eh_dominado[j] = True

    nao_dominadas = [candidatos[i] for i in range(n) if not eh_dominado[i]]
    return nao_dominadas


# ---------------------------------------------------------------------------
# 4. Métricas de Avaliação Multiobjetivo (Hipervolume, Espaçamento, Cobertura)
# ---------------------------------------------------------------------------

def calcular_espacamento(pontos_3d):
    """
    Calcula a métrica de Espaçamento de Schott (Spacing - S):
    S = sqrt( 1/(N - 1) * sum((d_i - d_medio)^2) )
    onde d_i é a distância euclidiana da solução i até sua vizinha mais próxima na frente.
    Mede a uniformidade da distribuição das soluções ao longo da fronteira.
    """
    if len(pontos_3d) < 2:
        return 0.0

    n = len(pontos_3d)
    d = []
    for i in range(n):
        pi = pontos_3d[i]
        min_dist = float("inf")
        for j in range(n):
            if i == j:
                continue
            pj = pontos_3d[j]
            dist = math.sqrt(sum((pi[k] - pj[k]) ** 2 for k in range(len(pi))))
            if dist < min_dist:
                min_dist = dist
        d.append(min_dist)

    d_medio = sum(d) / n
    var = sum((di - d_medio) ** 2 for di in d) / (n - 1)
    return math.sqrt(var)


def calcular_hipervolume_2d(pontos_2d, ref_point):
    """
    Calcula a área dominada por um conjunto de pontos 2D em relação a um ponto de referência
    ref_point = (r0, r1), onde cada ponto p satisfaz p[0] <= r0 e p[1] <= r1.
    """
    # Filtra pontos que não dominam o ponto de referência
    pts = [p for p in pontos_2d if p[0] <= ref_point[0] and p[1] <= ref_point[1]]
    if not pts:
        return 0.0

    # Ordena crescentemente por p[0]
    pts = sorted(pts, key=lambda p: (p[0], p[1]))

    area = 0.0
    altura_atual = ref_point[1]

    for p in pts:
        if p[1] < altura_atual:
            largura = ref_point[0] - p[0]
            delta_altura = altura_atual - p[1]
            area += largura * delta_altura
            altura_atual = p[1]

    return area


def calcular_hipervolume_3d(pontos_3d, ref_point):
    """
    Calcula o Hipervolume 3D exato dominado pela fronteira de Pareto em relação
    a um ponto de referência ref_point = (r0, r1, r2).
    Utiliza decomposição analítica por fatiamento ao longo do eixo f0.
    """
    # Filtra pontos que caem dentro do hipercubo de referência
    pts = [
        p for p in pontos_3d
        if p[0] <= ref_point[0] and p[1] <= ref_point[1] and p[2] <= ref_point[2]
    ]
    if not pts:
        return 0.0

    # Coleta todas as coordenadas f0 únicas e as ordena
    f0_coords = sorted(list(set(p[0] for p in pts)) + [ref_point[0]])

    volume_total = 0.0

    for i in range(len(f0_coords) - 1):
        x_atual = f0_coords[i]
        x_prox = f0_coords[i + 1]
        delta_x = x_prox - x_atual
        if delta_x <= 1e-12:
            continue

        # Pontos ativos nesta fatia: todos aqueles cujo f0 <= x_atual
        pts_ativos = [(p[1], p[2]) for p in pts if p[0] <= x_atual + 1e-12]
        if pts_ativos:
            area_2d = calcular_hipervolume_2d(pts_ativos, (ref_point[1], ref_point[2]))
            volume_total += delta_x * area_2d

    return volume_total


def calcular_cobertura_dois_conjuntos(conjunto_a, conjunto_b, tol=1e-9):
    """
    Métrica C(A, B) de Zitzler:
    Calcula a fração de soluções em B que são dominadas por ao menos uma solução em A.
    C(A, B) = |{b in B | existe a in A com a domina b}| / |B|.
    Retorna float em [0.0, 1.0].
    """
    if not conjunto_b:
        return 0.0

    dominados = 0
    for b in conjunto_b:
        for a in conjunto_a:
            if domina_solucao(a, b, tol=tol):
                dominados += 1
                break

    return float(dominados) / len(conjunto_b)
