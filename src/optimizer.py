import heapq
import itertools
from src.graph import ResourceGraph


class PathOptimizer:
    def __init__(self, graph: ResourceGraph, llm_scores: dict = None):
        self.graph = graph
        self.llm_scores = llm_scores or {}  # {rid: 0.0-1.0}

    def _llm_boost(self, rid: str) -> float:
        """Retorna el boost del LLM para un recurso (neutro 0.5 si no hay score)."""
        return self.llm_scores.get(rid, 0.5) * 20  # escala el score al rango del scoring

    # ------------------------------------------------------------------
    # Política de puntuación compartida (greedy y beam)
    # ------------------------------------------------------------------
    def _difficulty_jump_penalty(self, rid: str, selected: list) -> float:
        """
        Penaliza elegir un recurso mucho más difícil que lo visto hasta ahora en
        la ruta parcial 'selected'. Se permite subir un nivel de dificultad sin
        coste; cada nivel extra de salto resta 5 puntos.

        Multiplicador afinado empíricamente a 5 (un 15 degradaba la cobertura).
        Lo usan los tres algoritmos, de ahí que viva en un solo lugar.
        """
        resource_difficulty = self.graph.get_resource(rid).get("difficulty", 1)
        max_difficulty_so_far = self._get_max_difficulty(selected)
        return max(0, resource_difficulty - max_difficulty_so_far - 1) * 5

    def _score_resource(self, *, future: int, direct: int, hours: float,
                        rid: str, selected: list) -> float:
        """
        Política de puntuación común a greedy y beam_search (mayor = mejor).

        Centraliza los PESOS de la decisión, no las ENTRADAS: cada algoritmo
        calcula 'future', 'direct' y 'hours' a su manera (greedy razona en
        marginal por-recurso; beam en acumulado por-ruta) y aquí solo se combinan
        con el mismo criterio. Así se elimina la duplicación de la fórmula sin
        alterar el comportamiento de ninguno de los dos.

          - future : cobertura objetivo alcanzable por simulación hacia adelante
                     (lo más importante: resuelve la miopía del greedy puro)
          - direct : habilidades objetivo cubiertas de forma directa
          - hours  : coste temporal asociado a la decisión (se penaliza suave)
          - boost  : prior de relevancia del LLM (0-20)
          - pen.   : penalización por salto de dificultad
        """
        return (
            (future * 100)
            + (direct * 10)
            - (hours * 0.1)
            + self._llm_boost(rid)
            - self._difficulty_jump_penalty(rid, selected)
        )

    def _get_useful_candidates(self, known: frozenset, target: set) -> set:
        """
        Retorna los recursos que podrían formar parte de una ruta válida hacia el
        objetivo: los que enseñan habilidades objetivo pendientes MÁS, de forma
        TRANSITIVA, los que enseñan cualquier prerequisito (directo o indirecto)
        de esos recursos.

        Recorre la cadena COMPLETA de prerequisitos hasta el punto fijo, no solo
        un nivel. Cortar la cadena en un nivel dejaba al A* sin los prerequisitos
        profundos (p.ej. álgebra y cálculo bajo redes neuronales), provocando
        rutas vacías en objetivos que sí eran alcanzables.
        """
        remaining_target = target - known
        if not remaining_target:
            return set()

        useful = set()
        needed_skills = set(remaining_target)
        resolved_skills = set(known)

        while needed_skills:
            skill = needed_skills.pop()
            if skill in resolved_skills:
                continue
            resolved_skills.add(skill)
            for r in self.graph.get_resources_teaching(skill):
                useful.add(r["id"])
                for req in r["requires"]:
                    if req not in resolved_skills:
                        needed_skills.add(req)

        return useful if useful else set(self.graph.resources.keys())

    def _min_hours_to_cover(self, target_skills: set, known_skills: set = None,
                            max_hours: float = None):
        """
        Mínimo EXACTO de horas para cubrir target_skills, vía búsqueda de costo
        uniforme (Dijkstra) sobre el espacio de estados = conjunto de habilidades
        conocidas, con costo = horas. Como todas las aristas tienen costo >= 0, el
        primer estado que cubre el objetivo es óptimo en horas.

        Devuelve el mínimo de horas, o None si no se puede cubrir (dentro de
        max_hours, si se especifica). Se usa para la verificación de factibilidad,
        donde necesitamos un límite inferior fiable y no la heurística del A*.
        """
        known_base = frozenset(known_skills) if known_skills else frozenset()
        if target_skills.issubset(known_base):
            return 0.0

        counter = itertools.count()  # desempate para no comparar frozensets en el heap
        heap = [(0.0, next(counter), known_base)]
        best_cost = {known_base: 0.0}

        while heap:
            cost, _, known = heapq.heappop(heap)
            if cost > best_cost.get(known, float("inf")):
                continue
            if target_skills.issubset(known):
                return cost
            for rid, r in self.graph.resources.items():
                if not set(r["requires"]).issubset(known):
                    continue
                new_known = known | frozenset(r["teaches"])
                if new_known == known:
                    continue
                new_cost = cost + r["duration_hours"]
                if max_hours is not None and new_cost > max_hours:
                    continue
                if new_cost < best_cost.get(new_known, float("inf")):
                    best_cost[new_known] = new_cost
                    heapq.heappush(heap, (new_cost, next(counter), new_known))
        return None

    def _get_skills_from_resources(self, resource_ids: list) -> set:
        skills = set()
        for rid in resource_ids:
            r = self.graph.get_resource(rid)
            if r:
                skills.update(r["teaches"])
        return skills

    def _can_unlock(self, rid: str, current_skills: set) -> bool:
        r = self.graph.get_resource(rid)
        return set(r["requires"]).issubset(current_skills)

    def _get_max_difficulty(self, selected: list) -> int:
        """Retorna la dificultad máxima alcanzada en el path actual.
        Si el path está vacío, inicia en 1 para no penalizar recursos básicos."""
        if not selected:
            return 1
        return max(self.graph.get_resource(rid).get("difficulty", 1) for rid in selected)

    def _forward_coverage(self, known: set, selected: set,
                          target: set, max_hours: float,
                          hours_used: float) -> int:
        sim_known = known.copy()
        sim_hours = hours_used
        changed = True
        all_candidates = list(self.graph.resources.keys())
        while changed:
            changed = False
            for rid in all_candidates:
                if rid in selected:
                    continue
                r = self.graph.get_resource(rid)
                if not set(r["requires"]).issubset(sim_known):
                    continue
                if max_hours and (sim_hours + r["duration_hours"]) > max_hours:
                    continue
                new_skills = set(r["teaches"]) - sim_known
                if new_skills:
                    sim_known.update(new_skills)
                    sim_hours += r["duration_hours"]
                    changed = True
        return len(sim_known & target)

    # ------------------------------------------------------------------
    # Algoritmo 1: Greedy con lookahead
    # ------------------------------------------------------------------
    def greedy(self, target_skills: set, known_skills: set = None,
               max_hours: float = None) -> dict:

        known = set(known_skills) if known_skills else set()
        remaining = set(target_skills) - known
        selected = []
        total_hours = 0
        all_candidates = list(self.graph.resources.keys())

        iterations = 0
        max_iterations = len(all_candidates) * 3

        while remaining and iterations < max_iterations:
            iterations += 1
            best = None
            best_score = -1
            budget_left = (max_hours - total_hours) if max_hours else None

            for rid in all_candidates:
                if rid in selected:
                    continue
                if not self._can_unlock(rid, known):
                    continue

                r = self.graph.get_resource(rid)
                hours = r["duration_hours"]

                if budget_left and hours > budget_left:
                    continue

                new_skills = set(r["teaches"]) - known
                if not new_skills:
                    continue

                new_known = known | new_skills
                future = self._forward_coverage(
                    new_known, set(selected + [rid]),
                    target_skills, max_hours, total_hours + hours
                )
                # greedy razona en marginal: cobertura directa = skills nuevas
                # que entran en lo que aún falta; coste = horas de ESTE recurso.
                direct = len(new_skills & remaining)
                score = self._score_resource(
                    future=future, direct=direct, hours=hours,
                    rid=rid, selected=selected
                )

                if score > best_score:
                    best_score = score
                    best = rid

            if best is None:
                break

            r = self.graph.get_resource(best)
            selected.append(best)
            known.update(r["teaches"])
            remaining -= set(r["teaches"])
            total_hours += r["duration_hours"]

        ordered = self.graph.topological_sort(selected)
        covered = self._get_skills_from_resources(ordered)
        coverage = len(target_skills & covered) / len(target_skills) if target_skills else 0

        return {
            "algorithm": "greedy",
            "path": ordered,
            "total_hours": sum(
                self.graph.get_resource(r)["duration_hours"] for r in ordered
            ),
            "skills_covered": list(covered & target_skills),
            "skills_missing": list(target_skills - covered),
            "coverage_pct": round(coverage * 100, 1)
        }

    # ------------------------------------------------------------------
    # Algoritmo 2: Beam Search con forward simulation
    # ------------------------------------------------------------------
    def beam_search(self, target_skills: set, known_skills: set = None,
                    max_hours: float = None, beam_width: int = 4) -> dict:

        known_base = set(known_skills) if known_skills else set()
        all_candidates = list(self.graph.resources.keys())

        beams = [([], known_base.copy(), 0.0)]
        best_result = ([], known_base.copy(), 0.0)
        best_coverage = len(known_base & target_skills) / len(target_skills) if target_skills else 0

        for _ in range(len(all_candidates)):
            if not beams:
                break

            candidates = []

            for path, known, hours in beams:
                added_any = False

                for rid in all_candidates:
                    if rid in path:
                        continue
                    if not self._can_unlock(rid, known):
                        continue

                    r = self.graph.get_resource(rid)
                    new_hours = hours + r["duration_hours"]

                    if max_hours and new_hours > max_hours:
                        continue

                    new_skills = set(r["teaches"]) - known
                    if not new_skills:
                        continue

                    new_known = known | set(r["teaches"])
                    new_path = path + [rid]

                    future = self._forward_coverage(
                        new_known, set(new_path),
                        target_skills, max_hours, new_hours
                    )
                    # beam razona en acumulado: cobertura directa = todas las
                    # skills objetivo ya conocidas por la ruta; coste = horas
                    # acumuladas de toda la ruta.
                    direct = len(new_known & target_skills)
                    score = self._score_resource(
                        future=future, direct=direct, hours=new_hours,
                        rid=rid, selected=path
                    )

                    candidates.append((new_path, new_known, new_hours, score))
                    added_any = True

                if not added_any:
                    # Estado terminal del beam (no se puede extender): se puntúa
                    # solo por cobertura futura y coste, sin boost ni penalización
                    # (no hay recurso nuevo que puntuar). No pasa por
                    # _score_resource a propósito: su contrato espera un 'rid'.
                    future = self._forward_coverage(
                        known, set(path), target_skills, max_hours, hours
                    )
                    score = (future * 100) - (hours * 0.1)
                    candidates.append((path, known, hours, score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[3], reverse=True)
            beams = [(p, k, h) for p, k, h, _ in candidates[:beam_width]]

            for path, known, hours in beams:
                coverage = len(known & target_skills) / len(target_skills) if target_skills else 0
                if coverage > best_coverage or (
                    coverage == best_coverage and hours < best_result[2]
                ):
                    best_coverage = coverage
                    best_result = (path, known, hours)

            if best_coverage >= 1.0:
                break

        path, known, hours = best_result
        ordered = self.graph.topological_sort(path)
        covered = self._get_skills_from_resources(ordered)
        coverage = len(target_skills & covered) / len(target_skills) if target_skills else 0

        return {
            "algorithm": "beam_search",
            "path": ordered,
            "total_hours": sum(
                self.graph.get_resource(r)["duration_hours"] for r in ordered
            ),
            "skills_covered": list(covered & target_skills),
            "skills_missing": list(target_skills - covered),
            "coverage_pct": round(coverage * 100, 1)
        }

    # ------------------------------------------------------------------
    # Algoritmo 3: A* - búsqueda A*-inspirada con heurística aproximada
    # Estado: (horas_usadas, -cobertura, path, known_skills)
    # Heurística: habilidades objetivo aún faltantes desde el estado actual
    #
    # Nota: A* NO usa _score_resource. Su f = g + h·5 - boost - penalización es un
    # COSTO (menor = mejor), de estructura distinta al score de greedy/beam
    # (mayor = mejor). Forzarlo por el scorer común distorsionaría la búsqueda.
    # Solo comparte _llm_boost y _difficulty_jump_penalty.
    # ------------------------------------------------------------------
    def astar(self, target_skills: set, known_skills: set = None,
              max_hours: float = None) -> dict:

        known_base = frozenset(known_skills) if known_skills else frozenset()
        all_candidates_list = list(self.graph.resources.keys())

        def heuristic(known: frozenset) -> int:
            """Habilidades objetivo que aún faltan."""
            return len(target_skills - known)

        initial_h = heuristic(known_base)
        heap = [(initial_h, 0.0, 0, [], known_base)]

        visited = {}

        best_result = ([], known_base, 0.0)
        best_coverage = len(known_base & target_skills) / len(target_skills) if target_skills else 0

        iterations = 0
        max_iterations = max(5000, len(all_candidates_list) ** 3 * 5)

        while heap and iterations < max_iterations:
            iterations += 1
            _, hours, _, path, known = heapq.heappop(heap)

            state_key = known
            if state_key in visited and visited[state_key] <= hours:
                continue
            visited[state_key] = hours

            coverage = len(known & target_skills) / len(target_skills) if target_skills else 0
            if coverage > best_coverage or (
                coverage == best_coverage and hours < best_result[2]
            ):
                best_coverage = coverage
                best_result = (list(path), known, hours)

            if target_skills.issubset(known):
                break

            useful_candidates = self._get_useful_candidates(known, target_skills)

            for rid in all_candidates_list:
                if rid in path:
                    continue
                if rid not in useful_candidates:
                    continue
                if not self._can_unlock(rid, known):
                    continue

                r = self.graph.get_resource(rid)
                new_hours = hours + r["duration_hours"]

                if max_hours and new_hours > max_hours:
                    continue

                new_known = known | frozenset(r["teaches"])

                if new_known == known:
                    continue

                new_path = path + [rid]
                h = heuristic(new_known)
                direct = len(new_known & target_skills)

                # f = costo real + heurística - boost del LLM - penalización
                new_f = (new_hours + (h * 5)
                         - self._llm_boost(rid)
                         - self._difficulty_jump_penalty(rid, path))

                heapq.heappush(heap, (
                    new_f,
                    new_hours,
                    -direct,
                    new_path,
                    new_known
                ))

        path, known, hours = best_result
        ordered = self.graph.topological_sort(path)
        covered = self._get_skills_from_resources(ordered)
        coverage = len(target_skills & covered) / len(target_skills) if target_skills else 0

        return {
            "algorithm": "a_star",
            "path": ordered,
            "total_hours": sum(
                self.graph.get_resource(r)["duration_hours"] for r in ordered
            ),
            "skills_covered": list(covered & target_skills),
            "skills_missing": list(target_skills - covered),
            "coverage_pct": round(coverage * 100, 1)
        }

    def check_feasibility(self, target_skills: set, known_skills: set = None,
                      max_hours: float = None) -> dict:
        """
        Analiza si el objetivo es alcanzable antes de optimizar.
        Retorna un dict con: is_feasible, unreachable_skills, min_hours_needed,
        reachable_skills, y un mensaje explicativo.
        """
        known = set(known_skills) if known_skills else set()

        reachable = known.copy()
        changed = True
        while changed:
            changed = False
            for rid, r in self.graph.resources.items():
                new_skills = set(r["teaches"]) - reachable
                if new_skills and set(r["requires"]).issubset(reachable):
                    reachable.update(new_skills)
                    changed = True

        unreachable = target_skills - reachable
        reachable_targets = target_skills & reachable

        min_hours = self._min_hours_to_cover(reachable_targets, known_skills, max_hours=None)
        if min_hours is None:
            min_hours = 0

        hours_feasible = (max_hours is None) or (min_hours <= max_hours)
        fully_feasible = not unreachable and hours_feasible

        messages = []
        if unreachable:
            messages.append(
                f"Las habilidades {', '.join(unreachable)} no son alcanzables "
                f"con los recursos disponibles."
            )
        if not hours_feasible:
            messages.append(
                f"Cubrir las habilidades alcanzables requiere mínimo {min_hours}h "
                f"pero solo tienes {max_hours}h disponibles. "
                f"Se generará la mejor ruta posible dentro del presupuesto."
            )
        if fully_feasible:
            messages.append(
                f"Objetivo completamente alcanzable en {min_hours}h mínimo."
            )

        return {
            "is_feasible": fully_feasible,
            "unreachable_skills": list(unreachable),
            "reachable_skills": list(reachable_targets),
            "min_hours_needed": min_hours,
            "hours_available": max_hours,
            "message": " ".join(messages)
        }

    def compare(self, target_skills: set, known_skills: set = None,
                max_hours: float = None) -> dict:
        """
        Compara los tres algoritmos:
        - beam_width se calcula dinámicamente: max(3, min(6, |target_skills|))
          Asegura mínima exploración (3) sin ser excesivo (6)
        """
        beam_width = max(3, min(6, len(target_skills)))
        r1 = self.greedy(target_skills, known_skills, max_hours)
        r2 = self.beam_search(target_skills, known_skills, max_hours,
                              beam_width=beam_width)
        r3 = self.astar(target_skills, known_skills, max_hours)
        return {"greedy": r1, "beam_search": r2, "a_star": r3}