import json
from pathlib import Path


class ResourceGraph:
    def __init__(self, data_path: str = "data/resources.json") -> None:
        self.resources: dict[str, dict] = {}
        self.adjacency: dict[str, set[str]] = {}
        self._load(data_path)
        self._build_graph()

    def _load(self, path: str) -> None:
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data["resources"]:
            self.resources[r["id"]] = r

    def _build_graph(self) -> None:
        # Mapeo: habilidad → recursos que la enseñan
        skill_providers: dict[str, list[str]] = {}
        for rid, r in self.resources.items():
            for skill in r["teaches"]:
                skill_providers.setdefault(skill, []).append(rid)

        # Para cada recurso, encontrar sus prerequisitos directos
        for rid, r in self.resources.items():
            self.adjacency[rid] = set()
            for skill in r["requires"]:
                for provider in skill_providers.get(skill, []):
                    if provider != rid:
                        self.adjacency[rid].add(provider)

        # Validar que no hay ciclos en el grafo
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(
                f"Se detectaron ciclos en el grafo de dependencias: {cycles}. "
                "Revisa la estructura de prerequisitos en resources.json"
            )

    def get_resource(self, rid: str) -> dict | None:
        return self.resources.get(rid)

    def get_prerequisites(self, rid: str) -> set[str]:
        return self.adjacency.get(rid, set())

    def get_all_prerequisites(self, rid: str) -> set[str]:
        """Obtiene TODOS los prerequisitos transitivos de un recurso."""
        visited: set[str] = set()
        stack: list[str] = list(self.adjacency.get(rid, set()))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self.adjacency.get(current, set()))
        return visited

    def get_resources_teaching(self, skill: str) -> list[dict]:
        """Retorna todos los recursos que enseñan una habilidad dada."""
        return [
            r for r in self.resources.values()
            if skill in r["teaches"]
        ]

    def skills_by_domain(self) -> dict[str, list[str]]:
        """Agrupa las habilidades enseñadas por dominio (clave 'domain').

        Capa de datos para la acotación de dominio: el banner del catálogo (CLI y
        futura UI) deriva de aquí qué áreas cubre el sistema, sin nada hardcodeado.
        Las skills de cada dominio se devuelven ordenadas y sin duplicados.
        """
        grouped: dict[str, set[str]] = {}
        for r in self.resources.values():
            grouped.setdefault(r["domain"], set()).update(r["teaches"])
        return {d: sorted(skills) for d, skills in grouped.items()}

    def topological_sort(self, resource_ids: list[str]) -> list[str]:
        """Ordena una lista de recursos respetando sus dependencias."""
        ids = set(resource_ids)
        visited: set[str] = set()
        result: list[str] = []

        def visit(rid: str) -> None:
            if rid in visited:
                return
            visited.add(rid)
            for prereq in self.adjacency.get(rid, set()):
                if prereq in ids:
                    visit(prereq)
            result.append(rid)

        for rid in ids:
            visit(rid)

        return result

    def detect_cycles(self) -> list[list[str]]:
        """
        Detecta ciclos en el grafo de dependencias usando DFS.
        Retorna lista de ciclos encontrados, vacía si no hay ciclos.
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path[:])
                elif neighbor in rec_stack:
                    # Encontramos un ciclo
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node in self.resources.keys():
            if node not in visited:
                dfs(node, [])

        return cycles

    def summary(self) -> None:
        print(f"Recursos cargados : {len(self.resources)}")
        print(f"Relaciones totales: {sum(len(v) for v in self.adjacency.values())}")
        print()
        for rid, prereqs in self.adjacency.items():
            name = self.resources[rid]["name"]
            if prereqs:
                prereq_names = [self.resources[p]["name"] for p in prereqs]
                print(f"  {name}")
                for pn in prereq_names:
                    print(f"    ← {pn}")