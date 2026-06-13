"""
FuzzyPathOptimizer: el agente con politica de decision DIFUSA (modulo Simulacion).

AISLAMIENTO TOTAL (HANDOFF_SIMULACION.md §2): subclase de PathOptimizer que
sobreescribe UNICAMENTE `_score_resource`. La clase base queda intacta byte a byte;
el modulo de IA jamas ve esta subclase. Si el difuso tiene un bug, PathOptimizer
-y por tanto la IA- no se entera.

Por que basta con sobreescribir `_score_resource`:
  - greedy y beam_search llaman a `self._score_resource(...)`, asi que el override
    entra por POLIMORFISMO, sin duplicar su logica.
  - A* NO usa `_score_resource` (su f = g + h), asi que se HEREDA identico de la
    base y queda independiente del LLM. Esto preserva el invariante mas limpio del
    proyecto (A* base == A* difuso). (Verificado en tests/test_simulation.py.)
  - compare(), check_feasibility(), _min_hours_to_cover() se heredan sin tocar.

Los wrappers de greedy/beam_search solo capturan target/max_hours (que
`_score_resource` necesita para normalizar cobertura y horas) y delegan en super().
"""
from src.optimizer import PathOptimizer
from src.fuzzy_scorer import fuzzy_utility


class FuzzyPathOptimizer(PathOptimizer):
    def __init__(self, graph, llm_scores: dict[str, float] | None = None) -> None:
        super().__init__(graph, llm_scores)
        # H de respaldo para normalizar horas cuando no hay max_hours.
        self._total_catalog_hours = sum(
            r["duration_hours"] for r in graph.resources.values()
        ) or 1.0
        self._fz_target: set[str] = set()
        self._fz_max_hours: float | None = None

    # ------------------------------------------------------------------
    # Wrappers finos: capturan el contexto que el scoring difuso necesita
    # (|target| para cobertura, max_hours para horas) y delegan en la base.
    # ------------------------------------------------------------------
    def greedy(self, target_skills, known_skills=None, max_hours=None) -> dict:
        self._fz_target = set(target_skills)
        self._fz_max_hours = max_hours
        return super().greedy(target_skills, known_skills, max_hours)

    def beam_search(self, target_skills, known_skills=None, max_hours=None,
                    beam_width: int = 4) -> dict:
        self._fz_target = set(target_skills)
        self._fz_max_hours = max_hours
        return super().beam_search(target_skills, known_skills, max_hours, beam_width)

    # astar NO se sobreescribe -> heredado identico, independiente del LLM.

    # ------------------------------------------------------------------
    # Override de la POLITICA de scoring: misma firma que la base, pero la
    # combinacion lineal se reemplaza por inferencia difusa Mamdani.
    #
    # Igual que el lineal, esto centraliza la COMBINACION de pesos, no las
    # ENTRADAS: greedy sigue pasando cantidades marginales y beam acumuladas; el
    # difuso solo las combina. Se preserva el gating de `contributes` (si es False,
    # la relevancia entra como 0.5 neutro) para no contaminar la condicion OFF.
    # ------------------------------------------------------------------
    def _score_resource(self, *, future, direct, hours, rid, selected,
                        contributes: bool = True) -> float:
        # relevancia: la unica entrada "ruidosa" (score del LLM). Gating intacto.
        relevancia = self.llm_scores.get(rid, 0.5) if contributes else 0.5

        # cobertura objetivo alcanzable, normalizada a [0,1].
        n_target = max(1, len(self._fz_target))
        cobertura = min(1.0, future / n_target)

        # horas, normalizadas por el presupuesto (o el total del catalogo).
        H = self._fz_max_hours if (self._fz_max_hours and self._fz_max_hours > 0) \
            else self._total_catalog_hours
        horas = min(1.0, hours / H) if H > 0 else 0.0

        # salto de dificultad: mismo calculo crudo que _difficulty_jump_penalty
        # (sin el x5 del lineal), normalizado por 3 niveles.
        max_dif = self._get_max_difficulty(selected)
        res_dif = self.graph.get_resource(rid).get("difficulty", 1)
        salto = min(1.0, max(0, res_dif - max_dif - 1) / 3.0)

        return fuzzy_utility({
            "relevancia": relevancia,
            "cobertura": cobertura,
            "horas": horas,
            "salto": salto,
        })
