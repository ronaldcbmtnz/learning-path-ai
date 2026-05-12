from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient


def print_path(result: dict, graph: ResourceGraph):
    print(f"\n  Algoritmo : {result['algorithm'].upper()}")
    print(f"  Horas     : {result['total_hours']}h")
    print(f"  Cobertura : {result['coverage_pct']}%")
    print(f"\n  Ruta:")
    for i, rid in enumerate(result["path"], 1):
        r = graph.get_resource(rid)
        print(f"    {i}. {r['name']} ({r['duration_hours']}h, {r['type']})")
    if result["skills_missing"]:
        print(f"\n  Habilidades no cubiertas: {', '.join(result['skills_missing'])}")


def run():
    print("=" * 60)
    print("   GENERADOR DE RUTAS DE APRENDIZAJE CON IA")
    print("=" * 60)

    # Inicializar componentes
    graph = ResourceGraph()
    optimizer = PathOptimizer(graph)
    llm = LLMClient()

    # Paso 1: objetivo del usuario
    print("\nEscribe tu objetivo de aprendizaje en tus propias palabras.")
    print("Ejemplo: 'quiero aprender machine learning, se python basico y tengo 50 horas'\n")
    user_input = input("Tu objetivo: ").strip()

    if not user_input:
        print("No ingresaste ningún objetivo.")
        return

    # Paso 2: LLM parsea el objetivo conociendo las habilidades disponibles
    print("\nAnalizando tu objetivo...")
    available_skills = []
    for r in graph.resources.values():
        available_skills.extend(r["teaches"])
    available_skills = sorted(set(available_skills))

    parsed = llm.parse_user_goal(user_input, available_skills)

    print(f"\nObjetivo detectado  : {parsed['goal_summary']}")
    print(f"Habilidades meta    : {', '.join(parsed['target_skills'])}")
    print(f"Habilidades previas : {', '.join(parsed['known_skills']) or 'ninguna'}")
    print(f"Horas disponibles   : {parsed['max_hours'] or 'sin límite'}")

    target_skills = set(parsed["target_skills"])
    known_skills = set(parsed["known_skills"])
    max_hours = parsed["max_hours"]

    if not target_skills:
        print("\nNo se detectaron habilidades objetivo. Intenta ser más específico.")
        return

    # Paso 3: generar rutas con ambos algoritmos
    print("\nGenerando rutas de aprendizaje...")
    comparison = optimizer.compare(target_skills, known_skills, max_hours)

    greedy_result = comparison["greedy"]
    beam_result = comparison["beam_search"]

    print("\n" + "=" * 60)
    print("RUTAS GENERADAS")
    print("=" * 60)
    print_path(greedy_result, graph)
    print()
    print_path(beam_result, graph)

    # Paso 4: LLM compara y recomienda
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO (IA)")
    print("=" * 60)
    analysis = llm.compare_algorithms(greedy_result, beam_result, parsed["goal_summary"])
    print(f"\n  Recomendación : {analysis['recommended'].upper()}")
    print(f"  Razón         : {analysis['reason']}")
    print(f"  Trade-off     : {analysis['tradeoff']}")

    # Paso 5: LLM explica la ruta recomendada
    recommended = greedy_result if analysis["recommended"] == "greedy" else beam_result

    print("\n" + "=" * 60)
    print("EXPLICACIÓN DE TU RUTA")
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