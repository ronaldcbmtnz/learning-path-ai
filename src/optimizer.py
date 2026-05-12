from src.graph import ResourceGraph


class PathOptimizer:
    def __init__(self, graph: ResourceGraph):
        self.graph = graph

    def _get_skills_from_resources(self, resource_ids: list) -> set:
        """Retorna el conjunto de habilidades que entregan una lista de recursos."""
        skills = set()
        for rid in resource_ids:
            r = self.graph.get_resource(rid)
            if r:
                skills.update(r["teaches"])
        return skills

    def _can_unlock(self, rid: str, current_skills: set) -> bool:
        """Verifica si un recurso es accesible con las habilidades actuales."""
        r = self.graph.get_resource(rid)
        return set(r["requires"]).issubset(current_skills)

    def _covers_target(self, resource_ids: list, target_skills: set) -> bool:
        """Verifica si una ruta cubre todas las habilidades objetivo."""
        learned = self._get_skills_from_resources(resource_ids)
        return target_skills.issubset(learned)

    # ------------------------------------------------------------------
    # Algoritmo 1: Greedy
    # Estrategia: en cada paso elige el recurso disponible que enseña
    # más habilidades nuevas relevantes al objetivo.
    # ------------------------------------------------------------------
    def greedy(self, target_skills: set, known_skills: set = None,
               max_hours: float = None) -> dict:

        known = set(known_skills) if known_skills else set()
        remaining = set(target_skills)
        selected = []
        total_hours = 0

        # Incluir prerequisitos necesarios automáticamente
        all_candidates = list(self.graph.resources.keys())

        iterations = 0
        max_iterations = len(all_candidates) * 2

        while remaining and iterations < max_iterations:
            iterations += 1
            best = None
            best_score = -1

            for rid in all_candidates:
                if rid in selected:
                    continue
                if not self._can_unlock(rid, known):
                    continue

                r = self.graph.get_resource(rid)
                hours = r["duration_hours"]

                if max_hours and (total_hours + hours) > max_hours:
                    continue

                # Score: habilidades nuevas relevantes que aporta
                new_skills = set(r["teaches"]) & remaining
                score = len(new_skills)

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
            "total_hours": sum(self.graph.get_resource(r)["duration_hours"] for r in ordered),
            "skills_covered": list(covered & target_skills),
            "skills_missing": list(target_skills - covered),
            "coverage_pct": round(coverage * 100, 1)
        }

    # ------------------------------------------------------------------
    # Algoritmo 2: Beam Search
    # Estrategia: mantiene los K mejores caminos parciales en cada paso
    # en lugar de comprometerse con uno solo como greedy.
    # ------------------------------------------------------------------
    def beam_search(self, target_skills: set, known_skills: set = None,
                    max_hours: float = None, beam_width: int = 3) -> dict:

        known_base = set(known_skills) if known_skills else set()

        # Cada estado: (recursos_seleccionados, habilidades_conocidas, horas_usadas)
        beams = [([], known_base.copy(), 0.0)]
        best_result = None
        best_coverage = -1

        all_candidates = list(self.graph.resources.keys())

        for _ in range(len(all_candidates)):
            candidates = []

            for path, known, hours in beams:
                extended = False
                for rid in all_candidates:
                    if rid in path:
                        continue
                    if not self._can_unlock(rid, known):
                        continue

                    r = self.graph.get_resource(rid)
                    new_hours = hours + r["duration_hours"]

                    if max_hours and new_hours > max_hours:
                        continue

                    new_path = path + [rid]
                    new_known = known | set(r["teaches"])
                    covered = new_known & target_skills
                    score = len(covered) - (new_hours / 100)

                    candidates.append((new_path, new_known, new_hours, score))
                    extended = True

                if not extended and path:
                    covered = known & target_skills
                    score = len(covered) - (hours / 100)
                    candidates.append((path, known, hours, score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[3], reverse=True)
            beams = [(p, k, h) for p, k, h, _ in candidates[:beam_width]]

            # Guardar el mejor resultado hasta ahora
            for path, known, hours in beams:
                covered = known & target_skills
                coverage = len(covered) / len(target_skills) if target_skills else 0
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_result = (path, known, hours)

            if best_coverage == 1.0:
                break

        if best_result is None:
            best_result = ([], known_base, 0)

        path, known, hours = best_result
        ordered = self.graph.topological_sort(path)
        covered = self._get_skills_from_resources(ordered)
        coverage = len(target_skills & covered) / len(target_skills) if target_skills else 0

        return {
            "algorithm": "beam_search",
            "path": ordered,
            "total_hours": sum(self.graph.get_resource(r)["duration_hours"] for r in ordered),
            "skills_covered": list(covered & target_skills),
            "skills_missing": list(target_skills - covered),
            "coverage_pct": round(coverage * 100, 1)
        }

    def compare(self, target_skills: set, known_skills: set = None,
                max_hours: float = None) -> dict:
        """Ejecuta ambos algoritmos y retorna los dos resultados para comparar."""
        r1 = self.greedy(target_skills, known_skills, max_hours)
        r2 = self.beam_search(target_skills, known_skills, max_hours)
        return {"greedy": r1, "beam_search": r2}