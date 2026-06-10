from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient
from src import domain


def print_path(result: dict, graph: ResourceGraph) -> None:
    algo_labels = {
        "greedy": "GREEDY",
        "beam_search": "BEAM SEARCH",
        "a_star": " A* "
    }
    print(f"\n  Algoritmo : {algo_labels.get(result['algorithm'], result['algorithm'])}")
    print(f"  Horas     : {result['total_hours']}h")
    print(f"  Cobertura : {result['coverage_pct']}%")
    print(f"\n  Ruta:")
    for i, rid in enumerate(result["path"], 1):
        r = graph.get_resource(rid)
        print(f"    {i}. {r['name']} ({r['duration_hours']}h, {r['type']})")
    if result["skills_missing"]:
        print(f"\n  Habilidades no cubiertas: {', '.join(result['skills_missing'])}")


def run() -> None:
    print("=" * 60)
    print("   GENERADOR DE RUTAS DE APRENDIZAJE CON IA")
    print("=" * 60)

    graph = ResourceGraph()
    llm = LLMClient()                          # 1. llm primero

    # Acotación de dominio: el sistema es mono-dominio (tecnología). Mostrar el
    # alcance por adelantado evita que el usuario pida "a ciegas".
    print("\n" + domain.catalog_banner(graph))

    print("\nEscribe tu objetivo de aprendizaje en tus propias palabras.")
    print("Ejemplo: 'quiero aprender machine learning, se python basico y tengo 50 horas'\n")
    user_input = input("Tu objetivo: ").strip()

    if not user_input:
        print("No ingresaste ningún objetivo.")
        return

    # 2. Parsear objetivo → obtener parsed
    print("\nAnalizando tu objetivo...")
    available_skills = sorted(set(
        skill
        for r in graph.resources.values()
        for skill in r["teaches"]
    ))
    parsed = llm.parse_user_goal(user_input, available_skills)

    print(f"\nObjetivo detectado  : {parsed['goal_summary']}")
    print(f"Habilidades meta    : {', '.join(parsed['target_skills'])}")
    print(f"Habilidades previas : {', '.join(parsed['known_skills']) or 'ninguna'}")
    print(f"Horas disponibles   : {parsed['max_hours'] or 'sin límite'}")

    target_skills = set(parsed["target_skills"])
    known_skills = set(parsed["known_skills"])
    max_hours = parsed["max_hours"]

    if not target_skills:
        # Fuera de alcance: el LLM no pudo mapear el objetivo a ninguna habilidad
        # del catálogo (petición de otro dominio o demasiado vaga). No se generan
        # recursos; se explica el alcance.
        print("\n" + domain.out_of_scope_message(graph))
        return

    
    # 3. Ahora sí: evaluar relevancia con LLM y crear optimizer
    print("\nEvaluando relevancia de recursos con IA...")
    all_resources = list(graph.resources.values())
    llm_scores = llm.score_resources_for_goal(parsed["goal_summary"], all_resources)
    optimizer = PathOptimizer(graph, llm_scores=llm_scores)

    top_resources = sorted(llm_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nRecursos más relevantes según IA:")
    for rid, score in top_resources:
        r = graph.get_resource(rid)
        print(f"  {r['name']:<45} relevancia: {score:.2f}")

    print("\nVerificando factibilidad del objetivo...")
    feasibility = optimizer.check_feasibility(target_skills, known_skills, max_hours)
    print(f"\n  {feasibility['message']}")

    if feasibility["unreachable_skills"]:
        print(f"  Habilidades fuera del catálogo ignoradas: {', '.join(feasibility['unreachable_skills'])}")
        # Continuar solo con las habilidades alcanzables
        target_skills = set(feasibility["reachable_skills"])
        if not target_skills:
            # Todo el objetivo cayó fuera del catálogo: explicar el alcance en vez
            # de generar recursos sintéticos (fallback generativo descartado).
            print("\n" + domain.out_of_scope_message(graph))
            return

    if not feasibility["is_feasible"] and max_hours:
        print(f"  Horas mínimas necesarias : {feasibility['min_hours_needed']}h")
        print(f"  Horas disponibles        : {max_hours}h")
        print("  Se generará la mejor ruta posible dentro del presupuesto.\n")
    
    # 4. Generar rutas
    print("\nGenerando rutas de aprendizaje...")
    comparison = optimizer.compare(target_skills, known_skills, max_hours)

    greedy_result = comparison["greedy"]
    beam_result = comparison["beam_search"]
    astar_result = comparison["a_star"]

    print("\n" + "=" * 60)
    print("RUTAS GENERADAS")
    print("=" * 60)
    print_path(greedy_result, graph)
    print()
    print_path(beam_result, graph)
    print()
    print_path(astar_result, graph)

    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO (IA)")
    print("=" * 60)
    analysis = llm.compare_algorithms(
        greedy_result, beam_result, astar_result, parsed["goal_summary"]
    )
    print(f"\n  Recomendación : {analysis['recommended'].upper()}")
    print(f"  Razón         : {analysis['reason']}")
    print(f"  Trade-off     : {analysis['tradeoff']}")

    recommended = comparison[analysis["recommended"]]

    print("\n" + "=" * 60)
    print("EXPLICACIÓN DE TU RUTA RECOMENDADA")
    print("=" * 60)
    explanation = llm.explain_path(
        recommended,
        parsed["goal_summary"],
        graph.get_resource
    )
    print(f"\n{explanation}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run()