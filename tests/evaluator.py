import time
import json
from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient
from tests.test_cases import TEST_CASES


def run_evaluation():
    graph = ResourceGraph()
    llm = LLMClient()
    results = []
    all_resources = list(graph.resources.values())

    print("=" * 85)
    print("EVALUACIÓN COMPARATIVA DE ALGORITMOS")
    print("=" * 85)
    print(f"{'ID':<6} {'Perfil':<35} {'Alg':<12} {'Horas':>6} {'Cob%':>6} {'LLM':>3} {'t(ms)':>7}")
    print("-" * 85)

    for tc in TEST_CASES:
        # Generar scores de relevancia para este test case
        llm_used = False
        try:
            llm_scores = llm.score_resources_for_goal(tc["profile"], all_resources)
            llm_used = True
        except Exception:
            # Fallback silencioso si LLM falla
            llm_scores = {}
        
        optimizer = PathOptimizer(graph, llm_scores=llm_scores)
        llm_indicator = "Y" if llm_used else "N"
        
        for algo in ["greedy", "beam_search", "a_star"]:
            start = time.time()

            if algo == "greedy":
                result = optimizer.greedy(
                    tc["target_skills"], tc["known_skills"], tc["max_hours"]
                )
            elif algo == "beam_search":
                # beam_width = max(3, min(6, num_skills)) asegura:
                # - Mínimo 3 para evitar búsqueda demasiado estrecha
                # - Máximo 6 para no explorar excesivamente
                # - Proporcional a complejidad del problema (cantidad de habilidades objetivo)
                beam_width = max(3, min(6, len(tc["target_skills"])))
                result = optimizer.beam_search(
                    tc["target_skills"], tc["known_skills"],
                    tc["max_hours"], beam_width=beam_width
                )
            else:
                result = optimizer.astar(
                    tc["target_skills"], tc["known_skills"], tc["max_hours"]
                )

            elapsed_ms = round((time.time() - start) * 1000, 1)

            row = {
                "id": tc["id"],
                "profile": tc["profile"],
                "algorithm": algo,
                "hours": result["total_hours"],
                "coverage_pct": result["coverage_pct"],
                "resources_count": len(result["path"]),
                "skills_missing": result["skills_missing"],
                "time_ms": elapsed_ms,
                "llm_used": llm_used
            }
            results.append(row)

            profile_short = tc["profile"][:34]
            print(f"{tc['id']:<6} {profile_short:<35} {algo:<12} "
                  f"{result['total_hours']:>6} {result['coverage_pct']:>6} {llm_indicator:>3} "
                  f"{elapsed_ms:>7}")

    print("=" * 85)
    _print_summary(results)

    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResultados guardados en data/evaluation_results.json")

    return results


def _print_summary(results):
    greedy_rows = [r for r in results if r["algorithm"] == "greedy"]
    beam_rows   = [r for r in results if r["algorithm"] == "beam_search"]
    astar_rows  = [r for r in results if r["algorithm"] == "a_star"]

    def avg(lst, key):
        return round(sum(r[key] for r in lst) / len(lst), 1)

    print("\nRESUMEN ESTADÍSTICO")
    print("-" * 50)
    print(f"{'Métrica':<30} {'Greedy':>8} {'Beam':>8} {'A*':>8}")
    print("-" * 50)
    print(f"{'Cobertura promedio (%)':<30} "
          f"{avg(greedy_rows, 'coverage_pct'):>8} "
          f"{avg(beam_rows, 'coverage_pct'):>8} "
          f"{avg(astar_rows, 'coverage_pct'):>8}")
    print(f"{'Horas promedio':<30} "
          f"{avg(greedy_rows, 'hours'):>8} "
          f"{avg(beam_rows, 'hours'):>8} "
          f"{avg(astar_rows, 'hours'):>8}")
    print(f"{'Recursos promedio':<30} "
          f"{avg(greedy_rows, 'resources_count'):>8} "
          f"{avg(beam_rows, 'resources_count'):>8} "
          f"{avg(astar_rows, 'resources_count'):>8}")
    print(f"{'Tiempo promedio (ms)':<30} "
          f"{avg(greedy_rows, 'time_ms'):>8} "
          f"{avg(beam_rows, 'time_ms'):>8} "
          f"{avg(astar_rows, 'time_ms'):>8}")

    print(f"\nCasos con cobertura 100%:")
    for label, rows in [("Greedy", greedy_rows), ("Beam", beam_rows), ("A*", astar_rows)]:
        perfect = sum(1 for r in rows if r["coverage_pct"] == 100.0)
        print(f"  {label:<10}: {perfect}/{len(rows)}")


if __name__ == "__main__":
    run_evaluation()