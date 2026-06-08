"""
Tests de invariantes del optimizador (Fase 5).

Verifican PROPIEDADES estructurales que deben cumplirse siempre, no los números
concretos de una corrida (que dependen del LLM no determinista). Por eso todo se
ejecuta en la condición OFF (``llm_scores={}``, determinista).

Son agnósticos al dataset a propósito: actúan como red de seguridad cuando se
amplíe el catálogo en la Fase 4. Un recurso nuevo que introduzca un ciclo, rompa
la admisibilidad de A* o haga que un algoritmo desborde el presupuesto se caza
aquí, sin tener que releer todos los números a mano.

La fuente de verdad del óptimo en horas es ``_min_hours_to_cover`` (Dijkstra
exacto), NO la heurística de A*.

Reproducibilidad: conviene correr con ``PYTHONHASHSEED=0`` (como el evaluador)
para fijar el orden de iteración de los ``set``. Aun así estos tests asertan
propiedades, no órdenes exactos, así que son robustos al seed; el seed solo
afecta a empates internos.
"""
import pytest

from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from tests.test_cases import TEST_CASES

ALGORITHMS = ("greedy", "beam_search", "astar")
DATA_PATH = "data/resources.json"


@pytest.fixture(scope="module")
def graph():
    return ResourceGraph(DATA_PATH)


def _run(graph, algo, target, known, max_hours, llm_scores=None):
    opt = PathOptimizer(graph, llm_scores=llm_scores or {})
    return getattr(opt, algo)(target, known, max_hours)


def _feasible_cases():
    """Casos completamente factibles (objetivo alcanzable Y dentro del presupuesto).
    Se calcula una vez al importar para poder parametrizar."""
    g = ResourceGraph(DATA_PATH)
    feasible = []
    for tc in TEST_CASES:
        opt = PathOptimizer(g)
        feas = opt.check_feasibility(
            tc["target_skills"], tc["known_skills"], tc["max_hours"]
        )
        if feas["is_feasible"]:
            feasible.append(tc)
    return feasible


FEASIBLE_CASES = _feasible_cases()


# ----------------------------------------------------------------------
# Grafo
# ----------------------------------------------------------------------
def test_graph_has_no_cycles(graph):
    """El grafo de prerequisitos debe ser un DAG; si no, topological_sort y A*
    pierden sentido."""
    assert graph.detect_cycles() == []


# ----------------------------------------------------------------------
# Presupuesto (regresión del bug §8.1)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_path_within_budget_canonical(graph, algo, tc):
    """Ninguna ruta puede exceder max_hours en los casos canónicos."""
    r = _run(graph, algo, tc["target_skills"], tc["known_skills"], tc["max_hours"])
    assert r["total_hours"] <= tc["max_hours"]


def test_greedy_budget_regression_min_trigger(graph):
    """Disparador mínimo conocido del bug §8.1: con target {python_basico,
    estadistica} y max_hours=3, greedy elige r10 (3h) y agota el presupuesto
    (budget_left==0); el guard falsy antiguo dejaba añadir luego r01 (10h),
    desbordando a 13h. Tras el fix no debe exceder 3h."""
    opt = PathOptimizer(graph, llm_scores={})
    r = opt.greedy({"python_basico", "estadistica"}, set(), 3)
    assert r["total_hours"] <= 3


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_no_budget_overflow_sweep(graph, algo):
    """Barrido de presupuestos (incluido max_hours=0): el guard falsy
    ``if budget_left and ...`` / ``if max_hours and ...`` dejaba pasar recursos
    cuando el valor era 0. Ningún algoritmo debe desbordar para ningún
    presupuesto."""
    for tc in TEST_CASES:
        for mh in range(0, 130):
            r = _run(graph, algo, tc["target_skills"], tc["known_skills"], mh)
            assert r["total_hours"] <= mh, (
                f"{algo} {tc['id']} max_hours={mh} -> {r['total_hours']}h"
            )


# ----------------------------------------------------------------------
# Validez topológica de la ruta devuelta
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_path_respects_prerequisites(graph, algo, tc):
    """Recorriendo la ruta en orden, los requires de cada recurso deben estar
    cubiertos por known U los recursos anteriores."""
    r = _run(graph, algo, tc["target_skills"], tc["known_skills"], tc["max_hours"])
    acquired = set(tc["known_skills"])
    for rid in r["path"]:
        res = graph.get_resource(rid)
        assert set(res["requires"]).issubset(acquired), (
            f"{algo} {tc['id']}: {rid} requiere {res['requires']} "
            f"no cubierto por {acquired}"
        )
        acquired |= set(res["teaches"])


# ----------------------------------------------------------------------
# A* óptimo y techo de cobertura (en casos factibles)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tc", FEASIBLE_CASES, ids=lambda t: t["id"])
def test_astar_optimal_hours_when_feasible(graph, tc):
    """En casos factibles, A* debe cubrir el 100% y hacerlo en el mínimo exacto
    de horas (== _min_hours_to_cover, Dijkstra). Es el resultado más limpio del
    proyecto: A* = óptimo determinista, independiente del LLM."""
    opt = PathOptimizer(graph, llm_scores={})
    r = opt.astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
    min_hours = opt._min_hours_to_cover(
        set(tc["target_skills"]), set(tc["known_skills"])
    )
    assert r["coverage_pct"] == 100.0
    assert r["total_hours"] == min_hours


@pytest.mark.parametrize("tc", FEASIBLE_CASES, ids=lambda t: t["id"])
def test_astar_is_coverage_ceiling_when_feasible(graph, tc):
    """En casos factibles, greedy y beam no pueden superar la cobertura de A*
    (que ya topa al 100%)."""
    g_ = _run(graph, "greedy", tc["target_skills"], tc["known_skills"], tc["max_hours"])
    b_ = _run(graph, "beam_search", tc["target_skills"], tc["known_skills"], tc["max_hours"])
    a_ = _run(graph, "astar", tc["target_skills"], tc["known_skills"], tc["max_hours"])
    assert g_["coverage_pct"] <= a_["coverage_pct"]
    assert b_["coverage_pct"] <= a_["coverage_pct"]


# ----------------------------------------------------------------------
# Neutralidad / determinismo de la condición OFF y de A*
# ----------------------------------------------------------------------
def test_llm_boost_neutral_when_scores_empty(graph):
    """Con scores vacíos (condición OFF) el boost es 0 para todos los recursos,
    contribuyan o no. Si esto se rompe, la ablación queda contaminada."""
    opt = PathOptimizer(graph, llm_scores={})
    for rid in graph.resources:
        assert opt._llm_boost(rid, contributes=True) == 0.0
        assert opt._llm_boost(rid, contributes=False) == 0.0


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_off_condition_deterministic(graph, algo, tc):
    """La condición OFF debe ser perfectamente reproducible: dos corridas
    idénticas dan el mismo resultado."""
    r1 = _run(graph, algo, tc["target_skills"], tc["known_skills"], tc["max_hours"])
    r2 = _run(graph, algo, tc["target_skills"], tc["known_skills"], tc["max_hours"])
    assert r1 == r2


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda t: t["id"])
def test_astar_independent_of_llm(graph, tc):
    """A* no usa la señal del LLM por construcción (f = g + h). Su resultado con
    scores arbitrarios debe ser idéntico al de la condición OFF; si difieren,
    alguien volvió a meter el LLM en el coste (§7)."""
    off = PathOptimizer(graph, llm_scores={}).astar(
        tc["target_skills"], tc["known_skills"], tc["max_hours"]
    )
    biased_scores = {rid: 0.95 for rid in graph.resources}
    on = PathOptimizer(graph, llm_scores=biased_scores).astar(
        tc["target_skills"], tc["known_skills"], tc["max_hours"]
    )
    assert off == on
