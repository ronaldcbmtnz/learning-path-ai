"""
Sistema de inferencia difusa Mamdani para la utilidad del agente (modulo
Simulacion, Cap. 5). Implementado A MANO (sin scikit-fuzzy u otra libreria): son
~150 lineas, evita una dependencia y demuestra que se entienden los mecanismos.

Es el espejo difuso de la politica lineal `PathOptimizer._score_resource`:
combina las MISMAS senales (relevancia del LLM, cobertura futura, horas, salto de
dificultad) pero razonando con grados ("relevancia alta", "horas cortas") en vez
de una formula lineal rigida. Constituye un nuevo eje de ablacion: lineal vs difuso.

AISLAMIENTO (HANDOFF_SIMULACION.md §4): NO importa graph, optimizer ni streamlit.
Recibe rasgos ya normalizados en [0,1] y devuelve un numero (utilidad
defuzzificada). matplotlib se usa solo en src/sim_ui.py para graficar; aqui se
exponen las definiciones de las membresias para que esa capa las dibuje.

Mecanismos: membresias TRIANGULARES; AND = min; implicacion = min (recorte);
agregacion = max; defuzzificacion = CENTROIDE sobre el universo de salida.
"""

# ----------------------------------------------------------------------
# Variables linguisticas de ENTRADA (universo [0,1]) y SALIDA (universo [0,100]).
# Cada etiqueta es un triangulo (a, b, c). a==b => hombro izquierdo; b==c => derecho.
# ----------------------------------------------------------------------
INPUT_MFS: dict[str, dict[str, tuple[float, float, float]]] = {
    "relevancia": {  # score del LLM (la unica entrada "ruidosa")
        "baja":  (0.0, 0.0, 0.5),
        "media": (0.25, 0.5, 0.75),
        "alta":  (0.5, 1.0, 1.0),
    },
    "cobertura": {   # future / |target|  (lo mas importante, como en el lineal)
        "pobre":    (0.0, 0.0, 0.5),
        "moderada": (0.25, 0.5, 0.75),
        "rica":     (0.5, 1.0, 1.0),
    },
    "horas": {       # min(1, hours / H)  (coste; se penaliza)
        "cortas": (0.0, 0.0, 0.5),
        "medias": (0.25, 0.5, 0.75),
        "largas": (0.5, 1.0, 1.0),
    },
    "salto": {       # min(1, salto_crudo / 3)  (salto de dificultad)
        "suave":  (0.0, 0.0, 0.5),
        "brusco": (0.5, 1.0, 1.0),
    },
}

OUTPUT_MF: dict[str, tuple[float, float, float]] = {
    "muy_baja": (0.0, 0.0, 25.0),
    "baja":     (0.0, 25.0, 50.0),
    "media":    (25.0, 50.0, 75.0),
    "alta":     (50.0, 75.0, 100.0),
    "muy_alta": (75.0, 100.0, 100.0),
}

OUTPUT_RANGE: tuple[float, float] = (0.0, 100.0)

# ----------------------------------------------------------------------
# Base de reglas REDUCIDA y legible (HANDOFF §5.2.2).
#
# Enumerar TODAS las combinaciones daria 3(relev) x 3(cob) x 3(horas) x 2(salto)
# = 54 reglas (81 si 'salto' tuviera 3 etiquetas). Esa explosion combinatoria es
# un hallazgo del curso: se documenta en FULL_RULE_COUNT y se mitiga cubriendo
# solo los casos dominantes. La cobertura domina la decision (como el termino
# 100*future del lineal); relevancia modula; horas y salto penalizan.
#
# Cada regla: (antecedentes {var: etiqueta}  -> consecuente etiqueta_salida).
# Antecedentes combinados con AND (min). Variables ausentes = "no importa".
# ----------------------------------------------------------------------
RULES: list[tuple[dict[str, str], str]] = [
    # --- cobertura rica: el recurso aporta mucho al objetivo ---
    ({"cobertura": "rica", "horas": "cortas", "salto": "suave"}, "muy_alta"),
    ({"cobertura": "rica", "relevancia": "alta"}, "muy_alta"),
    ({"cobertura": "rica", "horas": "medias"}, "alta"),
    ({"cobertura": "rica", "horas": "largas"}, "alta"),
    ({"cobertura": "rica", "relevancia": "baja"}, "alta"),
    # --- cobertura moderada: relevancia desempata ---
    ({"cobertura": "moderada", "relevancia": "alta"}, "alta"),
    ({"cobertura": "moderada", "relevancia": "media"}, "media"),
    ({"cobertura": "moderada", "relevancia": "baja"}, "media"),
    ({"cobertura": "moderada", "horas": "largas"}, "baja"),
    # --- cobertura pobre: el recurso casi no aporta ---
    ({"cobertura": "pobre", "relevancia": "alta"}, "baja"),
    ({"cobertura": "pobre", "relevancia": "baja"}, "muy_baja"),
    ({"cobertura": "pobre", "horas": "largas"}, "muy_baja"),
    # --- penalizaciones transversales ---
    ({"salto": "brusco", "cobertura": "moderada"}, "baja"),
    ({"salto": "brusco", "cobertura": "pobre"}, "muy_baja"),
]

# Conteo para el informe (§5.2.2): reglas enumeradas vs reglas usadas.
LABELS_PER_INPUT = {v: len(m) for v, m in INPUT_MFS.items()}
FULL_RULE_COUNT = 1
for _n in LABELS_PER_INPUT.values():
    FULL_RULE_COUNT *= _n          # 3*3*3*2 = 54
REDUCED_RULE_COUNT = len(RULES)


# ----------------------------------------------------------------------
# Membresia triangular. Maneja hombros (a==b o b==c) sin division por cero.
# ----------------------------------------------------------------------
def trimf(x: float, a: float, b: float, c: float) -> float:
    # Flanco ascendente a->b. a==b => hombro izquierdo: 1 en cuanto x alcanza b.
    if a == b:
        rise = 1.0 if x >= b else 0.0
    else:
        rise = (x - a) / (b - a)
    # Flanco descendente b->c. b==c => hombro derecho: 1 mientras x no pase b.
    if b == c:
        fall = 1.0 if x <= b else 0.0
    else:
        fall = (c - x) / (c - b)
    return max(min(rise, fall), 0.0)


def membership(var: str, label: str, x: float) -> float:
    """Grado de pertenencia de x a la etiqueta de una variable de entrada."""
    a, b, c = INPUT_MFS[var][label]
    return trimf(x, a, b, c)


def _fuzzify(features: dict[str, float]) -> dict[str, dict[str, float]]:
    """De valores crudos en [0,1] a grados de pertenencia por etiqueta."""
    fz: dict[str, dict[str, float]] = {}
    for var, labels in INPUT_MFS.items():
        x = features.get(var, 0.5)
        fz[var] = {lab: trimf(x, *params) for lab, params in labels.items()}
    return fz


# ----------------------------------------------------------------------
# Inferencia Mamdani + defuzzificacion por centroide.
# ----------------------------------------------------------------------
_CENTROID_STEPS = 100  # resolucion del universo de salida [0,100] (101 muestras)


def _infer(features: dict[str, float]) -> tuple[float, dict]:
    fz = _fuzzify(features)

    # Fuerza de activacion de cada regla = min de sus antecedentes (AND).
    fired: list[tuple[float, str, dict[str, str]]] = []
    for antecedents, consequent in RULES:
        strength = min((fz[var][lab] for var, lab in antecedents.items()), default=0.0)
        if strength > 0.0:
            fired.append((strength, consequent, antecedents))

    lo, hi = OUTPUT_RANGE
    n = _CENTROID_STEPS
    num = 0.0
    den = 0.0
    agg_samples: list[float] = []
    for i in range(n + 1):
        y = lo + (hi - lo) * i / n
        # Agregacion = max sobre reglas del recorte (implicacion = min) de su salida.
        mu = 0.0
        for strength, consequent, _ in fired:
            a, b, c = OUTPUT_MF[consequent]
            mu = max(mu, min(strength, trimf(y, a, b, c)))
        agg_samples.append(mu)
        num += y * mu
        den += mu

    utility = (num / den) if den > 0 else 50.0  # sin reglas activas -> neutro
    trace = {
        "fuzzified": fz,
        "fired_rules": [
            {"strength": round(s, 4), "consequent": cons, "antecedents": ant}
            for s, cons, ant in sorted(fired, key=lambda t: -t[0])
        ],
        "aggregated": agg_samples,
        "utility": utility,
    }
    return utility, trace


def fuzzy_utility(features: dict[str, float]) -> float:
    """Utilidad defuzzificada (centroide) en [0,100]. Mayor = mejor.

    `features` con claves: relevancia, cobertura, horas, salto (todas en [0,1];
    las ausentes se toman como 0.5, neutro). Determinista: mismos rasgos =>
    misma salida (clave para la condicion OFF de la ablacion)."""
    return _infer(features)[0]


def fuzzy_utility_trace(features: dict[str, float]) -> tuple[float, dict]:
    """Como fuzzy_utility, pero devuelve tambien el trace de inferencia
    (fuzzificacion, reglas activadas, salida agregada) para la visualizacion."""
    return _infer(features)
