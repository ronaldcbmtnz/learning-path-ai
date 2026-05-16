import heapq
from src.graph import ResourceGraph


class PathOptimizer:
    def __init__(self, graph: ResourceGraph):
        self.graph = graph

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
                direct = len(new_skills & remaining)
                score = (future * 100) + (direct * 10) - (hours * 0.1)

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
                    direct = len(new_known & target_skills)
                    score = (future * 100) + (direct * 10) - (new_hours * 0.1)

                    candidates.append((new_path, new_known, new_hours, score))
                    added_any = True

                if not added_any and path:
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
    # Algoritmo 3: A* 
    # Garantiza la ruta óptima: máxima cobertura con mínimas horas.
    # Estado: (horas_usadas, -cobertura, path, known_skills)
    # Heurística: habilidades objetivo aún alcanzables desde el estado actual
    # ------------------------------------------------------------------
    def astar(self, target_skills: set, known_skills: set = None,
              max_hours: float = None) -> dict:

        known_base = frozenset(known_skills) if known_skills else frozenset()
        all_candidates = list(self.graph.resources.keys())

        def heuristic(known: frozenset) -> int:
            """Habilidades objetivo que aún faltan (admisible: nunca sobreestima)."""
            return len(target_skills - known)

        # Cola de prioridad: (f, horas, -cobertura_directa, path, known)
        # f = horas_usadas + heurística (minimizamos horas, maximizamos cobertura)
        initial_h = heuristic(known_base)
        heap = [(initial_h, 0.0, 0, [], known_base)]
        
        # Visitados: known_skills → mejor costo encontrado
        visited = {}

        best_result = ([], known_base, 0.0)
        best_coverage = len(known_base & target_skills) / len(target_skills) if target_skills else 0

        iterations = 0
        max_iterations = 5000

        while heap and iterations < max_iterations:
            iterations += 1
            f, hours, neg_direct, path, known = heapq.heappop(heap)

            # Clave de estado: habilidades conocidas (no el path exacto)
            state_key = known
            if state_key in visited and visited[state_key] <= hours:
                continue
            visited[state_key] = hours

            # Actualizar mejor resultado encontrado
            coverage = len(known & target_skills) / len(target_skills) if target_skills else 0
            if coverage > best_coverage or (
                coverage == best_coverage and hours < best_result[2]
            ):
                best_coverage = coverage
                best_result = (list(path), known, hours)

            # Si cubrimos todo el objetivo, terminamos
            if target_skills.issubset(known):
                break

            # Expandir vecinos
            for rid in all_candidates:
                if rid in path:
                    continue
                if not self._can_unlock(rid, known):
                    continue

                r = self.graph.get_resource(rid)
                new_hours = hours + r["duration_hours"]

                if max_hours and new_hours > max_hours:
                    continue

                new_known = known | frozenset(r["teaches"])

                # Solo expandir si aporta habilidades nuevas
                if new_known == known:
                    continue

                new_path = path + [rid]
                h = heuristic(new_known)
                direct = len(new_known & target_skills)

                # f = costo real + heurística
                # Usamos horas como costo y heurística de habilidades faltantes
                new_f = new_hours + (h * 5)

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

    def compare(self, target_skills: set, known_skills: set = None,
                max_hours: float = None) -> dict:
        beam_width = max(3, min(6, len(target_skills)))
        r1 = self.greedy(target_skills, known_skills, max_hours)
        r2 = self.beam_search(target_skills, known_skills, max_hours,
                              beam_width=beam_width)
        r3 = self.astar(target_skills, known_skills, max_hours)
        return {"greedy": r1, "beam_search": r2, "a_star": r3}