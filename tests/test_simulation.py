"""
Invariantes del modulo de Simulacion (HANDOFF_SIMULACION.md §9 paso 6).

Property-based y deterministas. Cubren:
  - A* identico base vs fuzzy (y A* independiente de los scores).
  - Condicion OFF neutra/determinista en la subclase difusa.
  - Sistema difuso: determinismo, hombros de trimf, ordenes sensatos, OFF neutro.
  - Generadores aleatorios: rangos y reproducibilidad por semilla.
  - Monte Carlo: reproducibilidad por semilla, A* varianza 0, sigma=0 = identidad.
  - NO-REGRESION de la IA: importar este modulo (que importa la subclase difusa)
    NO altera los resultados de PathOptimizer base (huella OFF intacta).

Correr:  PYTHONHASHSEED=0 python -m pytest tests/test_simulation.py -q
"""
import json
import hashlib
import statistics

import pytest

from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.sim_optimizer import FuzzyPathOptimizer
from src.fuzzy_scorer import (
    fuzzy_utility, trimf, FULL_RULE_COUNT, REDUCED_RULE_COUNT,
)
from src.random_gen import RandomGen
from src.simulation import simulate, perturb_scores, get_record
from tests.test_cases import TEST_CASES

# Huella OFF determinista de PathOptimizer base sobre los 19 casos (Paso 0).
# Si importar la simulacion alterara la IA, este hash cambiaria.
BASELINE_FINGERPRINT = "d2132bf4f2dba7dd"


@pytest.fixture(scope="module")
def graph():
    return ResourceGraph()


def _bw(target):
    return max(3, min(6, len(target)))


# ----------------------------------------------------------------------
# A* identico base vs fuzzy + independiente del LLM
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_astar_identical_base_vs_fuzzy(graph, tc):
    a = PathOptimizer(graph, {}).astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
    b = FuzzyPathOptimizer(graph, {}).astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
    assert (a["path"], a["total_hours"], a["coverage_pct"]) == \
           (b["path"], b["total_hours"], b["coverage_pct"])


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_fuzzy_astar_independent_of_scores(graph, tc):
    off = FuzzyPathOptimizer(graph, {}).astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
    biased = {rid: 0.95 for rid in graph.resources}
    on = FuzzyPathOptimizer(graph, biased).astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
    assert off == on


# ----------------------------------------------------------------------
# Condicion OFF neutra/determinista en la subclase difusa
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_fuzzy_off_deterministic(graph, tc):
    for algo in ("greedy", "beam_search"):
        r1 = getattr(FuzzyPathOptimizer(graph, {}), algo)(
            tc["target_skills"], tc["known_skills"], tc["max_hours"])
        r2 = getattr(FuzzyPathOptimizer(graph, {}), algo)(
            tc["target_skills"], tc["known_skills"], tc["max_hours"])
        assert r1 == r2


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_fuzzy_paths_respect_prerequisites(graph, tc):
    """Toda ruta difusa sigue siendo topologicamente valida y dentro de presupuesto."""
    for algo in ("greedy", "beam_search"):
        r = getattr(FuzzyPathOptimizer(graph, {}), algo)(
            tc["target_skills"], tc["known_skills"], tc["max_hours"])
        if tc["max_hours"] is not None:
            assert r["total_hours"] <= tc["max_hours"]
        acquired = set(tc["known_skills"])
        for rid in r["path"]:
            res = graph.get_resource(rid)
            assert set(res["requires"]).issubset(acquired)
            acquired |= set(res["teaches"])


# ----------------------------------------------------------------------
# Sistema difuso
# ----------------------------------------------------------------------
def test_trimf_shoulders():
    assert trimf(0.0, 0, 0, 0.5) == 1.0          # hombro izq, pico en 0
    assert abs(trimf(0.25, 0, 0, 0.5) - 0.5) < 1e-9
    assert trimf(0.6, 0, 0, 0.5) == 0.0
    assert trimf(1.0, 0.5, 1, 1) == 1.0          # hombro der, pico en 1
    assert abs(trimf(0.75, 0.5, 1, 1) - 0.5) < 1e-9
    assert abs(trimf(0.5, 0.25, 0.5, 0.75) - 1.0) < 1e-9


def test_fuzzy_utility_deterministic():
    f = {"relevancia": 0.7, "cobertura": 0.6, "horas": 0.3, "salto": 0.0}
    assert fuzzy_utility(f) == fuzzy_utility(dict(f))


def test_fuzzy_sensible_ordering():
    alto = fuzzy_utility({"relevancia": 0.9, "cobertura": 1.0, "horas": 0.1, "salto": 0.0})
    bajo = fuzzy_utility({"relevancia": 0.1, "cobertura": 0.0, "horas": 1.0, "salto": 1.0})
    assert alto > 70 and bajo < 30 and alto > bajo


def test_fuzzy_coverage_monotonic_nondecreasing():
    vals = [fuzzy_utility({"relevancia": 0.5, "cobertura": c, "horas": 0.3, "salto": 0.0})
            for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_fuzzy_rule_explosion_documented():
    # 3*3*3*2 = 54 reglas si se enumeran todas; la base reducida usa menos.
    assert FULL_RULE_COUNT == 54
    assert 0 < REDUCED_RULE_COUNT < FULL_RULE_COUNT


# ----------------------------------------------------------------------
# Generadores aleatorios
# ----------------------------------------------------------------------
def test_uniform_in_range():
    g = RandomGen(1)
    xs = [g.uniform(2, 5) for _ in range(5000)]
    assert all(2 <= x <= 5 for x in xs)


def test_triangular_range_and_mean():
    g = RandomGen(2)
    xs = [g.triangular_sym(0.2) for _ in range(20000)]
    assert all(-0.2 <= x <= 0.2 for x in xs)
    assert abs(statistics.mean(xs)) < 0.01          # centrada en 0


def test_rng_reproducible_by_seed():
    g1, g2 = RandomGen(7), RandomGen(7)
    assert [g1.u01() for _ in range(5)] == [g2.u01() for _ in range(5)]
    g3 = RandomGen(8)
    assert [g3.u01() for _ in range(5)] != [RandomGen(7).u01() for _ in range(5)]


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------
def _scores(graph, val=0.6):
    return {rid: val for rid in graph.resources}


def test_perturb_sigma0_identity(graph):
    base = _scores(graph)
    assert perturb_scores(base, 0.0, RandomGen(0)) == base


def test_simulate_reproducible_by_seed(graph):
    tc = next(t for t in TEST_CASES if t["id"] == "TC17")
    base = _scores(graph)
    kw = dict(sigmas=[0.0, 0.2], n_runs=25)

    def slim(res):
        return [{k: v for k, v in r.items() if "samples" not in k} for r in res["records"]]

    a = simulate(graph, base, tc["target_skills"], tc["known_skills"], tc["max_hours"], seed=0, **kw)
    b = simulate(graph, base, tc["target_skills"], tc["known_skills"], tc["max_hours"], seed=0, **kw)
    assert slim(a) == slim(b)


def test_simulate_astar_zero_variance(graph):
    tc = next(t for t in TEST_CASES if t["id"] == "TC12")
    res = simulate(graph, _scores(graph), tc["target_skills"], tc["known_skills"],
                   tc["max_hours"], sigmas=[0.0, 0.1, 0.3], n_runs=20, seed=0)
    for r in res["records"]:
        if r["algorithm"] == "astar":
            assert r["cov_var"] == 0.0 and r["hours_var"] == 0.0


def test_simulate_sigma0_matches_base(graph):
    """sigma=0 (sin ruido) reproduce el resultado determinista del optimizador."""
    tc = next(t for t in TEST_CASES if t["id"] == "TC02")
    scores = _scores(graph, 0.7)
    res = simulate(graph, scores, tc["target_skills"], tc["known_skills"],
                   tc["max_hours"], sigmas=[0.0], n_runs=10, seed=0)
    ref = PathOptimizer(graph, scores).greedy(
        tc["target_skills"], tc["known_skills"], tc["max_hours"])
    rec = get_record(res, 0.0, "lineal", "greedy")
    assert rec["cov_mean"] == ref["coverage_pct"]
    assert rec["cov_var"] == 0.0


# ----------------------------------------------------------------------
# NO-REGRESION de la IA: importar la simulacion no altera PathOptimizer base
# ----------------------------------------------------------------------
def test_ia_baseline_fingerprint_unchanged(graph):
    fp = {}
    for tc in TEST_CASES:
        for algo in ("greedy", "beam_search", "astar"):
            r = getattr(PathOptimizer(graph, {}), algo)(
                tc["target_skills"], tc["known_skills"], tc["max_hours"])
            fp[f"{tc['id']}_{algo}"] = {
                "path": r["path"], "h": r["total_hours"], "cov": r["coverage_pct"]}
    h = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    assert h == BASELINE_FINGERPRINT
