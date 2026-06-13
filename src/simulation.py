"""
Motor de simulacion Monte Carlo (modulo Simulacion, Caps. 1, 2, 4).

Modela el ENTORNO NO DETERMINISTA del agente: el LLM es "un asesor que cambia un
poco de opinion cada vez". En vez de llamarlo N veces (caro/imposible en free-tier),
se le pregunta UNA vez (snapshot en disco) y se simula matematicamente ese temblor
N veces, perturbando cada score con ruido triangular (transformada inversa,
src/random_gen.py). Se mide como se distribuyen cobertura y horas, y si la politica
de utilidad DIFUSA aguanta mejor el ruido que la LINEAL.

RQ-SIM: bajo percepcion ruidosa del LLM, ¿como se distribuyen los resultados del
agente, y es el difuso mas robusto que el lineal?

AISLAMIENTO: usa PathOptimizer (lineal) y FuzzyPathOptimizer (difuso) SIN
modificarlos; A* es el CONTROL determinista (no depende de los scores -> varianza 0).
Cero llamadas al LLM: consume el snapshot. Todo aleatorio pasa por un RNG semillado.
"""
import statistics

from src.optimizer import PathOptimizer
from src.sim_optimizer import FuzzyPathOptimizer
from src.random_gen import RandomGen

POLICIES = ("lineal", "difuso")
SEARCH_ALGOS = ("greedy", "beam_search")   # sensibles al LLM (varian con el ruido)
CONTROL_ALGO = "astar"                     # control: independiente del LLM


def perturb_scores(base_scores: dict[str, float], sigma: float,
                   rng: RandomGen) -> dict[str, float]:
    """Perturba cada score con ruido triangular en [-sigma, sigma], recortado a
    [0,1] (Cap. 6.1.1: entorno no determinista). sigma=0 => sin ruido (identidad)."""
    if sigma <= 0:
        return dict(base_scores)
    out: dict[str, float] = {}
    for rid, s in base_scores.items():
        e = rng.triangular_sym(sigma)
        out[rid] = min(1.0, max(0.0, s + e))
    return out


def _beam_width(target: set) -> int:
    return max(3, min(6, len(target)))


def _agg(samples: list[float]) -> tuple[float, float]:
    """Media y varianza MUESTRAL (Cap. 4.1). Varianza 0 si hay <2 muestras."""
    if not samples:
        return 0.0, 0.0
    m = statistics.mean(samples)
    v = statistics.variance(samples) if len(samples) > 1 else 0.0
    return m, v


def simulate(graph, base_scores: dict[str, float], target_skills: set,
             known_skills: set | None, max_hours: float | None,
             sigmas: list[float], n_runs: int = 200, seed: int = 0) -> dict:
    """Corre el Monte Carlo para un objetivo y su vector de scores base.

    Para cada sigma y cada corrida, perturba los scores y evalua greedy/beam bajo
    las politicas lineal y difusa; agrega media y varianza por (sigma, politica,
    algoritmo). A* se calcula UNA vez (determinista) y se reporta con varianza 0.

    Devuelve una estructura JSON-serializable con agregados y muestras crudas (para
    histogramas en la UI). Reproducible: misma semilla => mismos resultados.
    """
    target = set(target_skills)
    known = set(known_skills) if known_skills else set()
    rng = RandomGen(seed)
    bw = _beam_width(target)

    # min_hours y A* son deterministas (no dependen de los scores).
    base = PathOptimizer(graph, {})
    min_hours = base.check_feasibility(target, known, max_hours)["min_hours_needed"]
    astar_res = base.astar(target, known, max_hours)
    astar_cov, astar_hours = astar_res["coverage_pct"], astar_res["total_hours"]

    records: list[dict] = []
    for sigma in sigmas:
        acc = {(p, a): {"cov": [], "hours": []}
               for p in POLICIES for a in SEARCH_ALGOS}

        for _ in range(n_runs):
            scores = perturb_scores(base_scores, sigma, rng)
            opts = {"lineal": PathOptimizer(graph, scores),
                    "difuso": FuzzyPathOptimizer(graph, scores)}
            for policy, opt in opts.items():
                g = opt.greedy(target, known, max_hours)
                b = opt.beam_search(target, known, max_hours, beam_width=bw)
                acc[(policy, "greedy")]["cov"].append(g["coverage_pct"])
                acc[(policy, "greedy")]["hours"].append(g["total_hours"])
                acc[(policy, "beam_search")]["cov"].append(b["coverage_pct"])
                acc[(policy, "beam_search")]["hours"].append(b["total_hours"])

        for (policy, algo), d in acc.items():
            cov_m, cov_v = _agg(d["cov"])
            hrs_m, hrs_v = _agg(d["hours"])
            records.append({
                "sigma": sigma, "policy": policy, "algorithm": algo,
                "cov_mean": round(cov_m, 2), "cov_var": round(cov_v, 3),
                "hours_mean": round(hrs_m, 2), "hours_var": round(hrs_v, 3),
                "cov_samples": d["cov"], "hours_samples": d["hours"],
                "constant": False,
            })

        # A* (control determinista): mismo valor en ambas politicas, varianza 0.
        for policy in POLICIES:
            records.append({
                "sigma": sigma, "policy": policy, "algorithm": CONTROL_ALGO,
                "cov_mean": astar_cov, "cov_var": 0.0,
                "hours_mean": astar_hours, "hours_var": 0.0,
                "cov_samples": [astar_cov], "hours_samples": [astar_hours],
                "constant": True,
            })

    return {
        "sigmas": list(sigmas), "n_runs": n_runs, "seed": seed,
        "min_hours": min_hours, "n_target": len(target),
        "astar_cov": astar_cov, "astar_hours": astar_hours,
        "records": records,
    }


def get_record(result: dict, sigma: float, policy: str, algorithm: str) -> dict | None:
    """Helper para la UI/tests: recupera un registro por (sigma, politica, algo)."""
    for r in result["records"]:
        if r["sigma"] == sigma and r["policy"] == policy and r["algorithm"] == algorithm:
            return r
    return None


def robustness_curve(result: dict, policy: str, algorithm: str) -> dict[str, list[float]]:
    """Curva de robustez vs sigma para (politica, algoritmo): media y varianza de
    cobertura a lo largo de los sigmas. Es el grafico central de RQ-SIM."""
    sigmas, cov_mean, cov_var = [], [], []
    for sigma in result["sigmas"]:
        rec = get_record(result, sigma, policy, algorithm)
        if rec:
            sigmas.append(sigma)
            cov_mean.append(rec["cov_mean"])
            cov_var.append(rec["cov_var"])
    return {"sigmas": sigmas, "cov_mean": cov_mean, "cov_var": cov_var}
