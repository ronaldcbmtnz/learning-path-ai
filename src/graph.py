import json
from pathlib import Path


class ResourceGraph:
    def __init__(self, data_path: str = "data/resources.json"):
        self.resources = {}
        self.adjacency = {}
        self._load(data_path)
        self._build_graph()

    def _load(self, path: str):
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data["resources"]:
            self.resources[r["id"]] = r

    def _build_graph(self):
        # Mapeo: habilidad → recursos que la enseñan
        skill_providers = {}
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

    def get_resource(self, rid: str) -> dict:
        return self.resources.get(rid)

    def get_prerequisites(self, rid: str) -> set:
        return self.adjacency.get(rid, set())

    def get_all_prerequisites(self, rid: str) -> set:
        """Obtiene TODOS los prerequisitos transitivos de un recurso."""
        visited = set()
        stack = list(self.adjacency.get(rid, set()))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self.adjacency.get(current, set()))
        return visited

    def get_resources_teaching(self, skill: str) -> list:
        """Retorna todos los recursos que enseñan una habilidad dada."""
        return [
            r for r in self.resources.values()
            if skill in r["teaches"]
        ]

    def topological_sort(self, resource_ids: list) -> list:
        """Ordena una lista de recursos respetando sus dependencias."""
        ids = set(resource_ids)
        visited = set()
        result = []

        def visit(rid):
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

    def summary(self):
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