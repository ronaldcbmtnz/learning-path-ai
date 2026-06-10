"""
Acotación de dominio del planificador (capa de presentación).

El sistema es un planificador de rutas de aprendizaje *mono-dominio*: solo cubre
tecnología (desarrollo de software, datos e IA). Esta acotación reemplaza al
fallback generativo que se barajó en la Fase 4: en un planificador específico de
dominio, inventar recursos para temas ajenos (medicina, arte, idiomas...) sería
poco honesto y poco útil. Ante una petición fuera de alcance, el sistema lo dice
con claridad en vez de generar.

Este módulo NO depende de ``ResourceGraph`` (los helpers reciben el grafo como
argumento) para que tanto el CLI como la futura UI lo reutilicen sin acoplarse.
"""

DOMAIN_NAME: str = "Tecnología: Desarrollo de Software, Datos e IA"

DOMAIN_DESCRIPTION: str = (
    "El sistema resuelve la planificación de rutas de aprendizaje en el dominio "
    "de la tecnología: programación, desarrollo web, matemáticas para IA, machine "
    "learning, ingeniería de datos, DevOps y MLOps. Se asume que el usuario tiene "
    "interés en este campo."
)

# Etiquetas legibles para cada clave de 'domain' usada en data/resources.json.
DOMAIN_LABELS: dict[str, str] = {
    "programacion": "Programación",
    "web": "Desarrollo Web",
    "matematicas": "Matemáticas para IA",
    "ml": "Machine Learning e IA",
    "datos": "Datos e Ingeniería de Datos",
    "devops": "DevOps y Cloud",
}

# Ejemplos de áreas que el sistema NO cubre (para el mensaje de fuera de alcance).
OUT_OF_SCOPE_EXAMPLES: list[str] = [
    "medicina",
    "derecho",
    "arte y diseño",
    "idiomas",
    "humanidades",
]


def domain_label(domain_key: str) -> str:
    """Etiqueta legible de un dominio; si es desconocido, capitaliza la clave."""
    return DOMAIN_LABELS.get(domain_key, domain_key.replace("_", " ").capitalize())


def catalog_banner(graph) -> str:
    """
    Banner de bienvenida con el alcance del catálogo. Reutilizable por CLI y UI.

    Deriva las áreas y el conteo de recursos del grafo (no hay nada hardcodeado),
    de modo que crece automáticamente al ampliar ``data/resources.json``.

    Solo ASCII a propósito: el CLI se ejecuta en consolas Windows (cp1252) donde
    ``print`` de emojis lanza UnicodeEncodeError. La UI (Streamlit) renderiza su
    propia versión con iconos a partir de los datos estructurados de este módulo.
    """
    by_domain = graph.skills_by_domain()
    areas = ", ".join(domain_label(d) for d in sorted(by_domain))
    n_resources = len(graph.resources)
    n_skills = sum(len(s) for s in by_domain.values())

    return (
        "Bienvenido al Planificador de Rutas de Aprendizaje\n"
        + "-" * 53 + "\n"
        f"  Dominio  : {DOMAIN_NAME}\n"
        f"  Areas    : {areas}\n"
        f"  Catalogo : {n_resources} recursos, {n_skills} habilidades\n"
        "  Fuera de alcance: "
        f"{', '.join(OUT_OF_SCOPE_EXAMPLES[:-1])} ni {OUT_OF_SCOPE_EXAMPLES[-1]}."
    )


def out_of_scope_message(graph=None) -> str:
    """
    Mensaje informativo cuando la petición cae fuera del alcance del catálogo
    (ninguna habilidad objetivo mapeable, o habilidades inalcanzables).

    Reemplaza al descartado fallback generativo: no se inventan recursos, se
    explica el alcance. Si se pasa el grafo, lista las áreas reales disponibles.
    """
    lines = [
        "Tu objetivo parece quedar fuera del alcance de este planificador.",
        f"Solo cubrimos el dominio de {DOMAIN_NAME}.",
    ]
    if graph is not None:
        areas = ", ".join(domain_label(d) for d in sorted(graph.skills_by_domain()))
        lines.append(f"Áreas disponibles: {areas}.")
    lines.append(
        "No ofrecemos recursos de "
        f"{', '.join(OUT_OF_SCOPE_EXAMPLES[:-1])} ni {OUT_OF_SCOPE_EXAMPLES[-1]}. "
        "Reformula tu objetivo dentro de estas áreas."
    )
    return " ".join(lines)
