"""
Diseño experimental para el generador de rutas de aprendizaje.

Responde tres preguntas de investigación:
  RQ1 (Ablation): ¿El scoring del LLM cambia/mejora las rutas que produce el
                  optimizador? Se compara cada algoritmo con la señal del LLM
                  activa ("llm_on") y desactivada ("llm_off").
  RQ2 (Factibilidad): ¿Qué casos son realmente alcanzables? Se separan las
                  instancias factibles de las infactibles para que los promedios
                  no mezclen "el algoritmo falló" con "el objetivo era imposible".
  RQ3 (Eficiencia): ¿Qué tan buena es cada ruta por hora invertida, y cuánto
                  tarda cada algoritmo (tiempo mediano sobre varias corridas)?

Nota sobre la ablation:
  "llm_off" se implementa pasando llm_scores={}. En ese caso _llm_boost() devuelve
  un valor constante (0.5*20) para todos los recursos, lo cual es NEUTRO para el
  ranking: un offset igual en todos los candidatos no cambia el argmax ni el orden
  de la cola de prioridad. Por eso scores vacíos = "el optimizador no recibe señal
  del LLM", el control limpio de la ablation. No requiere tocar el optimizer.
"""

import time
import json
import statistics
from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient
from tests.test_cases import TEST_CASES

# Repeticiones para medir tiempo. Se reporta la MEDIANA (robusta a outliers del SO).
N_TIMING_RUNS = 5

ALGORITHMS = ["greedy", "beam_search", "a_star"]
CONDITIONS = ["llm_on", "llm_off"]


# ----------------------------------------------------------------------
# Ejecución de algoritmos
# ----------------------------------------------------------------------
def _run_algorithm(optimizer: PathOptimizer, algo: str, tc: dict) -> dict:
    """Ejecuta un algoritmo sobre un test case y devuelve su dict de resultado."""
    if algo == "greedy":
        return optimizer.greedy(
            tc["target_skills"], tc["known_skills"], tc["max_hours"]
        )
    elif algo == "beam_search":
        beam_width = max(3, min(6, len(tc["target_skills"])))
        return optimizer.beam_search(
            tc["target_skills"], tc["known_skills"],
            tc["max_hours"], beam_width=beam_width
        )
    else:  # a_star
        return optimizer.astar(
            tc["target_skills"], tc["known_skills"], tc["max_hours"]
        )


def _median_time_ms(optimizer: PathOptimizer, algo: str, tc: dict,
                    runs: int = N_TIMING_RUNS) -> float:
    """Tiempo mediano de ejecución en ms sobre 'runs' repeticiones."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _run_algorithm(optimizer, algo, tc)
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 3)


# ----------------------------------------------------------------------
# Clasificación de factibilidad (independiente del LLM y del algoritmo)
# ----------------------------------------------------------------------
def _classify_feasibility(graph: ResourceGraph, tc: dict) -> tuple:
    """
    Clasifica un test case en: 'factible', 'infactible_catalogo'
    (hay habilidades que ningún recurso puede enseñar) o
    'infactible_presupuesto' (alcanzables pero no dentro de max_hours).
    """
    optimizer = PathOptimizer(graph, llm_scores={})
    feas = optimizer.check_feasibility(
        tc["target_skills"], tc["known_skills"], tc["max_hours"]
    )
    if feas["unreachable_skills"]:
        label = "infactible_catalogo"
    elif not feas["is_feasible"]:
        label = "infactible_presupuesto"
    else:
        label = "factible"
    return label, feas


# ----------------------------------------------------------------------
# Experimento principal
# ----------------------------------------------------------------------
def run_evaluation():
    graph = ResourceGraph()
    llm = LLMClient()
    all_resources = list(graph.resources.values())
    results = []

    print("=" * 80)
    print("DISEÑO EXPERIMENTAL: ABLATION DEL LLM + ANÁLISIS POR FACTIBILIDAD")
    print("=" * 80)
    print(f"Algoritmos: {', '.join(ALGORITHMS)}")
    print(f"Condiciones: {', '.join(CONDITIONS)} (ablation de la señal del LLM)")
    print(f"Tiempo: mediana sobre {N_TIMING_RUNS} corridas por celda")
    print("=" * 80)

    for tc in TEST_CASES:
        feas_label, _ = _classify_feasibility(graph, tc)

        # Señal del LLM: una sola muestra cacheada -> benchmark reproducible.
        llm_scores = llm.score_resources_for_goal(tc["profile"], all_resources)
        # ¿El LLM aportó señal real? Si todos los scores son idénticos, o falló
        # (fallback uniforme 0.5) o el LLM consideró todo igual de relevante.
        llm_active = len(set(llm_scores.values())) > 1
        if not llm_active:
            print(f"  [aviso] {tc['id']}: el LLM devolvió scores uniformes "
                  f"(¿rate limit o fallback?). La ablation puede no ser informativa aquí.")

        optimizers = {
            "llm_on": PathOptimizer(graph, llm_scores=llm_scores),
            "llm_off": PathOptimizer(graph, llm_scores={}),
        }

        for cond in CONDITIONS:
            optimizer = optimizers[cond]
            for algo in ALGORITHMS:
                result = _run_algorithm(optimizer, algo, tc)
                t_ms = _median_time_ms(optimizer, algo, tc)

                hours = result["total_hours"]
                cov = result["coverage_pct"]
                efficiency = round(cov / hours, 2) if hours > 0 else 0.0

                results.append({
                    "id": tc["id"],
                    "profile": tc["profile"],
                    "feasibility": feas_label,
                    "condition": cond,
                    "algorithm": algo,
                    "hours": hours,
                    "coverage_pct": cov,
                    "efficiency": efficiency,
                    "resources_count": len(result["path"]),
                    "skills_missing": result["skills_missing"],
                    "time_ms": t_ms,
                    "llm_active": llm_active,
                })

    _print_per_case_overview(results)
    _print_ablation_summary(results)
    _print_feasibility_summary(results)
    _print_ablation_deltas(results)

    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResultados completos guardados en data/evaluation_results.json")

    return results


# ----------------------------------------------------------------------
# Reportes
# ----------------------------------------------------------------------
def _avg(rows: list, key: str) -> float:
    vals = [r[key] for r in rows]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _print_per_case_overview(results: list):
    """Vista por caso: factibilidad + cobertura con LLM ON de cada algoritmo."""
    print("\nVISTA POR CASO (cobertura % con LLM ON)")
    print("-" * 80)
    print(f"{'ID':<6} {'Factibilidad':<24} {'Greedy':>8} {'Beam':>8} {'A*':>8}")
    print("-" * 80)
    case_ids = sorted(set(r["id"] for r in results))
    for cid in case_ids:
        feas = next(r["feasibility"] for r in results if r["id"] == cid)
        covs = {}
        for algo in ALGORITHMS:
            row = next(r for r in results
                       if r["id"] == cid and r["algorithm"] == algo
                       and r["condition"] == "llm_on")
            covs[algo] = row["coverage_pct"]
        print(f"{cid:<6} {feas:<24} "
              f"{covs['greedy']:>8} {covs['beam_search']:>8} {covs['a_star']:>8}")


def _print_ablation_summary(results: list):
    """RQ1: impacto agregado del LLM por algoritmo (todas las instancias)."""
    print("\n" + "=" * 80)
    print("RQ1 — ABLATION: IMPACTO DEL LLM (promedio sobre las 14 instancias)")
    print("=" * 80)
    print(f"{'Algoritmo':<14} {'Cond':<9} {'Cob%':>7} {'Horas':>7} "
          f"{'Efic':>7} {'100%':>7} {'t(ms)':>8}")
    print("-" * 80)
    for algo in ALGORITHMS:
        for cond in CONDITIONS:
            rows = [r for r in results
                    if r["algorithm"] == algo and r["condition"] == cond]
            perfect = sum(1 for r in rows if r["coverage_pct"] == 100.0)
            print(f"{algo:<14} {cond:<9} "
                  f"{_avg(rows, 'coverage_pct'):>7} "
                  f"{_avg(rows, 'hours'):>7} "
                  f"{_avg(rows, 'efficiency'):>7} "
                  f"{f'{perfect}/{len(rows)}':>7} "
                  f"{_avg(rows, 'time_ms'):>8}")
        print("-" * 80)


def _print_feasibility_summary(results: list):
    """RQ2: desempeño separando casos factibles de infactibles (LLM ON)."""
    print("\n" + "=" * 80)
    print("RQ2 — DESEMPEÑO POR FACTIBILIDAD (condición LLM ON)")
    print("=" * 80)
    for feas in ["factible", "infactible_presupuesto", "infactible_catalogo"]:
        subset = [r for r in results
                  if r["feasibility"] == feas and r["condition"] == "llm_on"]
        if not subset:
            continue
        n_cases = len(set(r["id"] for r in subset))
        print(f"\n{feas.upper()}  ({n_cases} casos)")
        print(f"  {'Algoritmo':<14} {'Cob. prom%':>11} {'Resueltos 100%':>16}")
        for algo in ALGORITHMS:
            rows = [r for r in subset if r["algorithm"] == algo]
            perfect = sum(1 for r in rows if r["coverage_pct"] == 100.0)
            print(f"  {algo:<14} {_avg(rows, 'coverage_pct'):>11} "
                  f"{f'{perfect}/{len(rows)}':>16}")


def _print_ablation_deltas(results: list):
    """
    RQ1 (cualitativo): casos concretos donde activar el LLM cambió la ruta.
    Si la lista sale vacía, es un hallazgo en sí mismo: con el peso actual la
    señal del LLM solo rompe empates y el término de cobertura la domina.
    """
    print("\n" + "=" * 80)
    print("RQ1 — CASOS DONDE EL LLM CAMBIÓ LA RUTA (ON vs OFF)")
    print("=" * 80)
    found = False
    case_ids = sorted(set(r["id"] for r in results))
    for cid in case_ids:
        for algo in ALGORITHMS:
            on = next(r for r in results if r["id"] == cid
                      and r["algorithm"] == algo and r["condition"] == "llm_on")
            off = next(r for r in results if r["id"] == cid
                       and r["algorithm"] == algo and r["condition"] == "llm_off")
            if (on["coverage_pct"] != off["coverage_pct"]
                    or on["hours"] != off["hours"]
                    or on["resources_count"] != off["resources_count"]):
                found = True
                print(f"  {cid} {algo:<12} "
                      f"ON: {on['coverage_pct']:>5}% {on['hours']:>3}h {on['resources_count']}r | "
                      f"OFF: {off['coverage_pct']:>5}% {off['hours']:>3}h {off['resources_count']}r")
    if not found:
        print("  Ninguna ruta cambió. Con el peso actual el boost del LLM (0-20)")
        print("  queda por debajo del término de cobertura (future*100), así que solo")
        print("  actúa como desempate. Esto motiva re-ponderar la señal en la Fase 2.")


if __name__ == "__main__":
    run_evaluation()