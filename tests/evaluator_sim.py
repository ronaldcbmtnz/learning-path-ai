"""
Experimento de SIMULACION (modulo Simulacion). Paralelo a tests/evaluator.py pero
en archivo aparte: NO ensucia el resultado mas limpio del modulo de IA.

Eje extra {lineal, difuso} x Monte Carlo sobre el ruido del LLM. Lee el snapshot
de scores (data/llm_scores_snapshot.json) -> CERO llamadas al LLM en vivo. Guarda
data/sim_results.json. Reporta media+-varianza y la curva de robustez vs sigma.

RQ-SIM: bajo ruido creciente en la percepcion del LLM, ¿cae menos / varia menos la
politica DIFUSA que la LINEAL? A* es el control determinista (varianza 0).

Solo ASCII en los prints (consola Windows cp1252). Reproducible con semilla fija.

Uso:  PYTHONHASHSEED=0 python -m tests.evaluator_sim
"""
import json
import statistics

from src.graph import ResourceGraph
from src.simulation import simulate, POLICIES, SEARCH_ALGOS, CONTROL_ALGO
from tests.test_cases import TEST_CASES

SNAPSHOT_PATH = "data/llm_scores_snapshot.json"
RESULTS_PATH = "data/sim_results.json"

SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_RUNS = 100
SEED = 0
ALGOS = (*SEARCH_ALGOS, CONTROL_ALGO)


def _load_snapshot() -> dict:
    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"No existe {SNAPSHOT_PATH}. Crealo primero:\n"
            "  python -m tools.build_llm_snapshot"
        )


def _strip_samples(result: dict) -> dict:
    """Quita las muestras crudas para que el JSON guardado sea compacto."""
    slim = {k: v for k, v in result.items() if k != "records"}
    slim["records"] = [
        {k: v for k, v in r.items() if k not in ("cov_samples", "hours_samples")}
        for r in result["records"]
    ]
    return slim


def run_simulation_evaluation():
    snapshot = _load_snapshot()
    graph = ResourceGraph()
    tc_by_id = {tc["id"]: tc for tc in TEST_CASES}

    print("=" * 78)
    print("EXPERIMENTO DE SIMULACION: ablacion lineal/difuso + Monte Carlo del ruido")
    print("=" * 78)
    print(f"Casos: {len(snapshot)} | sigmas: {SIGMAS} | N por celda: {N_RUNS} | semilla: {SEED}")
    print("Ruido: triangular en [-sigma, sigma] sobre los scores del LLM (snapshot).")
    print("=" * 78)

    per_case: dict[str, dict] = {}
    # agregados entre casos: (sigma, policy, algo) -> medias de cov y de varianza
    agg: dict[tuple, dict[str, list]] = {}

    for tc_id in sorted(snapshot):
        tc = tc_by_id[tc_id]
        base_scores = snapshot[tc_id]["scores"]
        res = simulate(graph, base_scores, tc["target_skills"],
                       tc["known_skills"], tc["max_hours"],
                       SIGMAS, n_runs=N_RUNS, seed=SEED)
        per_case[tc_id] = _strip_samples(res)
        for r in res["records"]:
            key = (r["sigma"], r["policy"], r["algorithm"])
            agg.setdefault(key, {"cov_mean": [], "cov_var": []})
            agg[key]["cov_mean"].append(r["cov_mean"])
            agg[key]["cov_var"].append(r["cov_var"])

    _print_sanity_sigma0(agg)
    _print_robustness(agg)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"sigmas": SIGMAS, "n_runs": N_RUNS, "seed": SEED,
                   "per_case": per_case}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {RESULTS_PATH}")
    return per_case


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 2) if xs else 0.0


def _print_sanity_sigma0(agg: dict):
    """Sanity sigma=0 (sin ruido): cobertura media por politica y algoritmo.
    Debe reproducir el resultado determinista del snapshot (ON)."""
    print("\nSANITY sigma=0 (sin ruido) - cobertura media entre casos (%)")
    print("-" * 78)
    print(f"  {'algoritmo':<14}{'lineal':>10}{'difuso':>10}")
    for algo in ALGOS:
        lin = _mean(agg[(0.0, 'lineal', algo)]["cov_mean"])
        fz = _mean(agg[(0.0, 'difuso', algo)]["cov_mean"])
        print(f"  {algo:<14}{lin:>10}{fz:>10}")


def _print_robustness(agg: dict):
    """Curva de robustez: cobertura media y varianza media (entre casos) al subir
    sigma. RQ-SIM: ¿el difuso cae menos / varia menos que el lineal?"""
    print("\n" + "=" * 78)
    print("CURVA DE ROBUSTEZ vs sigma (cobertura media % | varianza media intra-caso)")
    print("=" * 78)
    for algo in ALGOS:
        print(f"\n[{algo}]" + ("   (control: independiente del LLM)" if algo == CONTROL_ALGO else ""))
        header = "  sigma  " + "".join(f"{p:>22}" for p in POLICIES)
        print(header)
        for sigma in SIGMAS:
            cells = ""
            for policy in POLICIES:
                cm = _mean(agg[(sigma, policy, algo)]["cov_mean"])
                cv = _mean(agg[(sigma, policy, algo)]["cov_var"])
                cells += f"{f'{cm:.1f}% (var {cv:.1f})':>22}"
            print(f"  {sigma:<6}{cells}")

    # Resumen RQ-SIM: degradacion media de cobertura del sigma min al max.
    print("\n" + "-" * 78)
    print("RESUMEN RQ-SIM (greedy/beam): caida de cobertura media y varianza media")
    s0, sN = SIGMAS[0], SIGMAS[-1]
    for algo in SEARCH_ALGOS:
        print(f"\n  [{algo}]  de sigma={s0} a sigma={sN}:")
        for policy in POLICIES:
            cov0 = _mean(agg[(s0, policy, algo)]["cov_mean"])
            covN = _mean(agg[(sN, policy, algo)]["cov_mean"])
            varN = _mean(agg[(sN, policy, algo)]["cov_var"])
            print(f"    {policy:<8} cobertura {cov0:.1f}% -> {covN:.1f}% "
                  f"(caida {cov0 - covN:+.1f}pts) | varianza@{sN}={varN:.1f}")


if __name__ == "__main__":
    run_simulation_evaluation()
