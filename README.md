# Learning Path AI

AI-powered personalized learning-path generator that combines graph search /
optimization algorithms with a large language model to turn a natural-language
goal into a valid, hour-budgeted sequence of learning resources that respects
prerequisite dependencies.

> University AI final project (Topic 9 — "Building learning paths"). Domain-scoped
> planner for **Technology: Software Development, Data & AI**. The LLM is a
> *functional* component (not decorative): it interprets the goal and scores
> resource relevance, and an ablation experiment measures exactly what it adds.

## What it does

The user describes a learning goal in plain language. The system then:

1. **LLM** parses the goal → `target_skills`, `known_skills`, `max_hours`
   (constrained to the catalog; near-miss skill names are snapped to the catalog,
   and the time budget is robustly extracted from the text).
2. **LLM** scores each resource's relevance to the goal (0–1).
3. Builds a **DAG** of resources from their prerequisites (cycle-checked).
4. Runs **three search algorithms** — Greedy (with lookahead), Beam Search, and
   A\* — to produce candidate sequences.
5. **LLM** compares the three routes and recommends one.
6. **LLM** writes a short, motivating explanation of the recommended path.

Out-of-domain requests (medicine, art, languages…) are **not** answered with
invented resources: the system states its scope and declines (`src/domain.py`).

## Example

Input: *"I want to learn machine learning from scratch. I have about 80 hours."*

```
Algorithm : A*  (optimal in hours)
Coverage  : 100%
Hours     : 48h
Path:
  1. Probability & basic statistics (6h)
  2. Visual linear algebra (4h)
  3. Python for beginners (10h)
  4. Applied statistics with Python (8h)
  5. Introduction to Machine Learning (20h)
```

Note how A\* covers the goal in the **minimum** number of hours — it picks the cheap
math resources (6h + 4h) instead of the broader 15h "Maths for ML" course.

## Tech stack

- **Python 3.13** (native 3.10+ typing).
- **OpenRouter API** via the `openai` SDK — **free-tier models only, zero cost**
  (`openai/gpt-oss-120b:free`, `qwen/qwen3-coder:free`), with automatic model
  rotation on rate-limit (429) and SHA-256 response caching.
- **Custom graph engine** — prerequisite DAG, cycle detection, topological sort.
- **Three search algorithms** — Greedy + lookahead, Beam Search, and an
  **admissible & consistent A\*** (provably hour-optimal).
- **Streamlit UI** (dark "tech/AI" theme) + interactive CLI.

## Algorithms

**Greedy (with lookahead)** — at each step picks the resource that maximizes
forward-simulated target coverage, with a soft hours term and a difficulty-jump
penalty. Fast but myopic: fails some deep-chain goals that A\* solves.

**Beam Search** — keeps the K best partial paths (dynamic width 3–6), exploring
more of the space before committing. Same scoring policy as Greedy (accumulated
form).

**A\*** — `f = g + h`, with `g` = accumulated hours and an **admissible &
consistent** heuristic: *the max over the still-missing target skills of the
cheapest resource (in hours) that teaches it*. The LLM is deliberately kept **out**
of the cost (any extra term breaks admissibility), so A\* is **independent of the
LLM by construction**. Its hour-optimality is verified to match an exact
uniform-cost (Dijkstra) search on every feasible case.

### Benchmark — LLM ablation (19 test cases)

The central experiment runs each algorithm with the LLM signal **on** vs **off**.
The OFF column is deterministic (official baseline); the ON column is a
representative snapshot (the free-tier LLM is non-deterministic across runs).

| Algorithm   | Coverage OFF | Coverage ON | 100% OFF | 100% ON | Optimality gap (feasible, 100%) |
|-------------|:-----------:|:-----------:|:--------:|:-------:|:-------------------------------:|
| Greedy      | 43.0%       | 60.5%       | 6/19     | 9/19    | avg +2.4h / max +17h            |
| Beam Search | 48.2%       | 71.1%       | 7/19     | 11/19   | avg +7.6h / max +50h            |
| **A\***     | **84.2%**   | **84.2%**   | **14/19**| **14/19** | **0 (14/14 optimal)**         |

**LLM lift:** Greedy **+17.5 pts**, Beam **+22.9 pts**, A\* **+0**.
Feasibility of the 19 cases: **14 feasible, 5 infeasible-by-budget, 0
infeasible-by-catalog**.

**Key finding.** A\* already tops 100% on feasible cases and ignores the LLM, so the
LLM cannot improve it. The LLM acts as a **relevance prior that compensates the
myopia of Greedy/Beam** (rescuing several 0%→100% deep-chain cases) — but it is
**double-edged**: on scattered goals under a tight budget it can *lower* coverage
(e.g. Beam 100%→50%), because semantic relevance diverges from concrete skill
coverage. This is reported honestly as a finding, not hidden.

## Dataset

Hand-authored catalog (not LLM-generated): **48 resources, 141 skills, 6
sub-domains** (programming, web, maths-for-AI, ML, data/data-engineering, DevOps),
with deep dependency chains (e.g. `rag ← llms ← transformer ← neural-nets ←
supervised-ML`). Validity is enforced and tested: acyclic (`detect_cycles() == []`)
and no unreachable in-domain skill. Each resource has `id`, `name`, `type`,
`domain`, `duration_hours`, `difficulty` (1–3), `teaches`, `requires`.

## Project structure

```
learning-path-ai/
├── IA.py                       # Streamlit UI (entry point)
├── data/
│   ├── resources.json          # Catalog: 48 resources / 141 skills / 6 domains
│   └── evaluation_results.json # LLM-ablation benchmark output
├── src/
│   ├── graph.py                # ResourceGraph: DAG, cycle detection, topo-sort
│   ├── optimizer.py            # PathOptimizer: Greedy, Beam, A* + helpers
│   ├── llm_client.py           # LLMClient: OpenRouter, caching, rotation, fallbacks
│   ├── domain.py               # Domain scoping (mono-domain) + out-of-scope message
│   ├── main.py                 # Interactive CLI pipeline
│   └── ui.py                   # Pure HTML/CSS/Markdown builders for the UI theme
├── pages/
│   └── 2_Simulacion.py         # Simulation module page (isolated; see below)
├── tests/
│   ├── test_cases.py           # 19 user profiles (TC01–TC19)
│   ├── test_invariants.py      # 224 property-based invariants (deterministic)
│   ├── evaluator.py            # LLM-ablation experiment
│   └── …                       # simulation tests / evaluator
└── requirements*.txt           # runtime / dev / simulation deps
```

## How to run

```bash
# 1. Install runtime dependencies (zero-cost LLM via OpenRouter free tier)
python -m pip install -r requirements.txt
#    For the test suite:           python -m pip install -r requirements-dev.txt
#    For the simulation module:     python -m pip install -r requirements-sim.txt

# 2. Configure the LLM (get a free key at openrouter.ai)
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-oss-120b:free, qwen/qwen3-coder:free
EOF

# 3. Run
python -m src.main          # interactive CLI
streamlit run IA.py         # web UI (Streamlit)
```

The Streamlit UI shows the visible catalog, clickable example prompts, structured
controls (skill multiselects + hours slider), a feasibility badge, the 3-route
comparison, the LLM recommendation/explanation, a step-by-step timeline, Markdown
export, and a live **LLM on/off ablation** toggle.

## Evaluation & testing

```bash
# Reproducible invariants (acyclicity, budget, topo-validity, A* optimality,
# OFF determinism, A* LLM-independence). 224 tests.
PYTHONHASHSEED=0 python -m pytest tests/test_invariants.py -q

# LLM-ablation benchmark (writes data/evaluation_results.json)
PYTHONHASHSEED=0 python -m tests.evaluator
```

The invariant suite is **property-based and dataset-agnostic**: it asserts
structural properties (in the deterministic OFF condition), so it acts as a safety
net for any catalog change.

## Simulation module (extension)

An additional, **fully isolated** module (for a Simulation course) layers a
**fuzzy-logic** decision policy (a hand-built Mamdani inference system) and a
**Monte-Carlo** simulation that models the LLM's non-determinism as a random
variable, to measure the agent's robustness to noise. A\* serves as the
zero-variance control (it ignores the LLM). It lives in new files only
(`src/fuzzy_scorer.py`, `src/sim_optimizer.py`, `src/random_gen.py`,
`src/simulation.py`, `pages/2_Simulacion.py`, …) and **does not modify the AI
module**: the AI behavior is byte-for-byte unchanged whether the simulation works
or not.

```bash
python -m pip install -r requirements-sim.txt
PYTHONHASHSEED=0 python -m pytest tests/test_simulation.py -q   # 89 invariants
```

## License

MIT License — see the `LICENSE` file for details.
