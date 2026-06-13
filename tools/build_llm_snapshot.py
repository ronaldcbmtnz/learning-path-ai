"""
Utilidad de un solo uso (modulo de Simulacion): congela a disco una muestra de los
scores reales del LLM, para que la simulacion Monte Carlo sea reproducible SIN
volver a llamar al LLM (HANDOFF_SIMULACION.md, paso 1 / 6.3).

- Usa las MISMAS llamadas cacheadas (SHA256) que tests/evaluator.py: por cada
  TEST_CASE, score_resources_for_goal(profile, recursos). No introduce llamadas
  distintas a las que el modulo de IA ya hace.
- Idempotente: si data/llm_scores_snapshot.json ya existe, NO vuelve a llamar al
  LLM (a menos que se pase --force).
- AISLAMIENTO: solo LEE del modulo de IA (graph, llm_client, test_cases). No
  modifica nada. Solo ASCII en los prints (consola Windows cp1252).



Uso:  python -m tools.build_llm_snapshot [--force]
"""
import os
import sys
import json

from src.graph import ResourceGraph
from src.llm_client import LLMClient
from tests.test_cases import TEST_CASES

SNAPSHOT_PATH = "data/llm_scores_snapshot.json"


def build(force: bool = False) -> dict:
    if os.path.exists(SNAPSHOT_PATH) and not force:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)
        print(f"[skip] {SNAPSHOT_PATH} ya existe ({len(snap)} casos). "
              "Usa --force para regenerar.")
        return snap

    graph = ResourceGraph()
    llm = LLMClient()
    all_resources = list(graph.resources.values())

    snapshot: dict[str, dict] = {}
    print(f"Capturando scores del LLM para {len(TEST_CASES)} casos...")
    for tc in TEST_CASES:
        scores = llm.score_resources_for_goal(tc["profile"], all_resources)
        # Aviso de honestidad: scores uniformes => fallback (rate limit) o LLM
        # considero todo igual. Se guarda igual; la simulacion lo tolera.
        uniform = len(set(scores.values())) <= 1
        flag = "  [aviso: scores uniformes]" if uniform else ""
        snapshot[tc["id"]] = {
            "profile": tc["profile"],
            "scores": {rid: float(scores[rid]) for rid in sorted(scores)},
        }
        print(f"  {tc['id']}: {len(scores)} scores{flag}")

    os.makedirs("data", exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {SNAPSHOT_PATH} ({len(snapshot)} casos).")
    return snapshot


if __name__ == "__main__":
    build(force="--force" in sys.argv)
