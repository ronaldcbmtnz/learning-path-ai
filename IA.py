"""
UI Streamlit del Planificador de Rutas de Aprendizaje (Fase 6).

Capa de PRESENTACIÓN: orquesta el mismo backend que el CLI (`ResourceGraph`,
`PathOptimizer`, `LLMClient`, `domain`) sin duplicar lógica de negocio. Todo el
"look" vive en `src/ui.py`; la acotación de dominio en `src/domain.py`.

Ejecutar:  streamlit run IA.py
"""
import json

import streamlit as st

from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient
from src import domain, ui
from tests.test_cases import TEST_CASES

st.set_page_config(
    page_title="Rutas de Aprendizaje · IA",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.css(), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Recursos cacheados (singletons entre reruns)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_graph() -> ResourceGraph:
    return ResourceGraph()


@st.cache_resource(show_spinner=False)
def get_llm() -> LLMClient:
    return LLMClient()


graph = get_graph()
llm = get_llm()
ALL_SKILLS = sorted({s for r in graph.resources.values() for s in r["teaches"]})
BY_DOMAIN = graph.skills_by_domain()


# ----------------------------------------------------------------------
# Pipeline (reutiliza el backend; las llamadas LLM están cacheadas en el cliente)
# ----------------------------------------------------------------------
def resolve_goal(mode: str, nl_text: str, sel_target: list[str],
                 sel_known: list[str], sel_hours: int | None) -> dict | None:
    """Devuelve {goal_summary, target, known, max_hours} o None si no hay objetivo."""
    if mode == "nl":
        if not nl_text.strip():
            return None
        parsed = llm.parse_user_goal(nl_text, ALL_SKILLS)
        return {
            "goal_summary": parsed.get("goal_summary") or nl_text,
            "target": set(parsed.get("target_skills") or []),
            "known": set(parsed.get("known_skills") or []),
            "max_hours": parsed.get("max_hours"),
        }
    # modo controles: sin parseo del LLM
    if not sel_target:
        return None
    goal = "Aprender " + ", ".join(sel_target)
    if sel_known:
        goal += " (ya sé " + ", ".join(sel_known) + ")"
    return {"goal_summary": goal, "target": set(sel_target),
            "known": set(sel_known), "max_hours": sel_hours}


def classify(feas: dict) -> str:
    if feas["unreachable_skills"]:
        return "infactible_catalogo"
    return "factible" if feas["is_feasible"] else "infactible_presupuesto"


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧭 Rutas de Aprendizaje")
    st.caption(domain.DOMAIN_NAME)
    st.divider()

    use_llm = st.toggle(
        "Señal del LLM en la búsqueda", value=True,
        help="Ablación: ON pasa los scores del LLM al optimizador (prior de "
             "relevancia); OFF usa solo la estructura del grafo. A* es independiente "
             "del LLM por construcción.",
    )
    st.caption("🔬 Experimento central del proyecto: observa el efecto on/off.")
    st.divider()

    st.markdown("#### 📚 Catálogo disponible")
    st.caption("Sobre estas áreas puedes diseñar tu ruta:")
    with st.container(height=320):
        st.markdown(ui.domain_catalog(BY_DOMAIN, domain.DOMAIN_LABELS),
                    unsafe_allow_html=True)
    st.divider()
    st.caption("Fuera de alcance: " + ", ".join(domain.OUT_OF_SCOPE_EXAMPLES) + ".")
    st.caption("Gasto cero · modelos free-tier vía OpenRouter.")


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    ui.hero(domain.DOMAIN_NAME, len(graph.resources), len(ALL_SKILLS), len(BY_DOMAIN)),
    unsafe_allow_html=True,
)
st.write("")


# ----------------------------------------------------------------------
# Entrada del objetivo
# ----------------------------------------------------------------------
def _set_example(text: str) -> None:
    st.session_state["nl_input"] = text


BENCH_ALGOS = ["greedy", "beam_search", "a_star"]
ALGO_NAMES = {"greedy": "Greedy", "beam_search": "Beam Search", "a_star": "A*"}
FEAS_ICON = {"factible": "✅ factible",
             "infactible_presupuesto": "🟠 presupuesto",
             "infactible_catalogo": "🔴 catálogo"}


@st.cache_data(show_spinner=False)
def _load_score_snapshot() -> dict | None:
    """Scores reales del LLM congelados a disco (data/llm_scores_snapshot.json).
    Permiten la condición ON sin llamar al LLM -> banco de pruebas reproducible."""
    try:
        with open("data/llm_scores_snapshot.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _run_benchmark(snapshot: dict | None) -> list[dict]:
    """Corre los 19 casos × 3 algoritmos en ON (scores del snapshot) y OFF (sin
    señal). Reutiliza PathOptimizer; CERO llamadas nuevas al LLM (reproducible).
    Es el experimento de tests/evaluator.py llevado a la UI."""
    graph = get_graph()
    rows: list[dict] = []
    for tc in TEST_CASES:
        feas = PathOptimizer(graph, {}).check_feasibility(
            tc["target_skills"], tc["known_skills"], tc["max_hours"])
        if feas["unreachable_skills"]:
            label = "infactible_catalogo"
        elif not feas["is_feasible"]:
            label = "infactible_presupuesto"
        else:
            label = "factible"
        min_hours = feas["min_hours_needed"]
        scores_on = (snapshot or {}).get(tc["id"], {}).get("scores", {})
        opts = {"on": PathOptimizer(graph, scores_on),
                "off": PathOptimizer(graph, {})}
        bw = max(3, min(6, len(tc["target_skills"])))
        for cond, opt in opts.items():
            for algo in BENCH_ALGOS:
                if algo == "beam_search":
                    r = opt.beam_search(tc["target_skills"], tc["known_skills"],
                                        tc["max_hours"], beam_width=bw)
                elif algo == "a_star":
                    r = opt.astar(tc["target_skills"], tc["known_skills"], tc["max_hours"])
                else:  # greedy
                    r = opt.greedy(tc["target_skills"], tc["known_skills"], tc["max_hours"])
                cov, hrs = r["coverage_pct"], r["total_hours"]
                gap = (hrs - min_hours) if (cov == 100.0 and label == "factible") else None
                rows.append({
                    "id": tc["id"], "profile": tc["profile"],
                    "target": ", ".join(sorted(tc["target_skills"])),
                    "feasibility": label, "condition": cond, "algorithm": algo,
                    "coverage": cov, "hours": hrs, "resources": len(r["path"]),
                    "path": list(r["path"]), "gap": gap, "min_hours": min_hours,
                })
    return rows


def _bench_get(rows: list[dict], tc_id: str, algo: str, cond: str) -> dict:
    return next(r for r in rows if r["id"] == tc_id
               and r["algorithm"] == algo and r["condition"] == cond)


tab_nl, tab_ctrl, tab_tests = st.tabs(
    ["💬  Lenguaje natural", "🎛️  Controles precisos", "🧪  Tests automatizados"]
)

with tab_nl:
    st.markdown(ui.section_header("¿Qué quieres aprender?", "💬"), unsafe_allow_html=True)
    st.caption("Descríbelo con tus palabras; el LLM extrae tus metas, lo que ya "
               "sabes y tus horas. O prueba un ejemplo:")
    ex_cols = st.columns(len(ui.EXAMPLE_PROMPTS))
    for col, ex in zip(ex_cols, ui.EXAMPLE_PROMPTS):
        col.button(f"{ex['icon']}  {ex['title']}", use_container_width=True,
                   on_click=_set_example, args=(ex["text"],), key=f"ex_{ex['title']}")
    nl_text = st.text_area(
        "Tu objetivo", key="nl_input", height=110, label_visibility="collapsed",
        placeholder="Ej.: Quiero aprender deep learning y NLP. Sé Python y algo de "
                    "ML. Tengo unas 60 horas.",
    )
    go_nl = st.button("🚀  Generar mi ruta", type="primary", use_container_width=True,
                      key="go_nl")
    # La ruta se renderiza DENTRO de esta pestaña (no debajo de todas), así no
    # aparece bajo la pestaña de tests al hacer scroll.
    nl_results = st.container()

with tab_ctrl:
    st.markdown(ui.section_header("Arma tu objetivo a mano", "🎛️"), unsafe_allow_html=True)
    st.caption("Sin depender del parseo del LLM: elige habilidades meta, lo que ya "
               "dominas y tus horas.")
    sel_target = st.multiselect("🎯 Habilidades que quieres aprender", ALL_SKILLS,
                                key="sel_target")
    sel_known = st.multiselect("✅ Habilidades que ya tienes", ALL_SKILLS,
                               key="sel_known")
    sel_hours = st.slider("⏱️ Horas disponibles", 0, 400, 60, step=5, key="sel_hours")
    go_ctrl = st.button("🚀  Generar mi ruta", type="primary",
                        use_container_width=True, key="go_ctrl")
    ctrl_results = st.container()

with tab_tests:
    st.markdown(ui.section_header("Banco de pruebas — 19 perfiles × 3 algoritmos", "🧪"),
                unsafe_allow_html=True)
    st.caption("Genera la ruta de cada perfil (TC01–TC19) con greedy, beam y A*, en "
               "condición LLM ON (scores congelados) y OFF (sin señal), y resume la "
               "ablación. Es el experimento de tests/evaluator.py en vivo, sin "
               "llamadas nuevas al LLM (reproducible).")
    snap = _load_score_snapshot()
    if snap is None:
        st.warning("No se encontró `data/llm_scores_snapshot.json` (scores del LLM "
                   "congelados): la columna **ON** saldrá igual que **OFF**. Genéralo "
                   "con `python -m tools.build_llm_snapshot`.")
    if st.button("▶  Ejecutar banco de pruebas (19 casos)", type="primary",
                 use_container_width=True, key="run_bench"):
        with st.spinner("Generando 19 casos × 3 algoritmos × ON/OFF…"):
            st.session_state["bench"] = _run_benchmark(snap)

    rows = st.session_state.get("bench")
    if rows is None:
        st.info("Pulsa **Ejecutar banco de pruebas** para correr los 19 casos.")
    else:
        n_cases = len({r["id"] for r in rows})
        case_ids = sorted({r["id"] for r in rows})

        def _avg(rs, k):
            v = [x[k] for x in rs]
            return round(sum(v) / len(v), 1) if v else 0.0

        # --- RQ1: ablación del LLM ---
        st.markdown("#### Ablación del LLM (RQ1) — ON vs OFF")
        abl = []
        for algo in BENCH_ALGOS:
            on = [r for r in rows if r["algorithm"] == algo and r["condition"] == "on"]
            off = [r for r in rows if r["algorithm"] == algo and r["condition"] == "off"]
            p_on = sum(1 for r in on if r["coverage"] == 100.0)
            p_off = sum(1 for r in off if r["coverage"] == 100.0)
            abl.append({
                "Algoritmo": ALGO_NAMES[algo],
                "Cob% OFF": _avg(off, "coverage"), "Cob% ON": _avg(on, "coverage"),
                "Lift (pts)": round(_avg(on, "coverage") - _avg(off, "coverage"), 1),
                "100% OFF": f"{p_off}/{n_cases}", "100% ON": f"{p_on}/{n_cases}",
            })
        st.dataframe(abl, hide_index=True, use_container_width=True)
        st.caption("A* es independiente del LLM por construcción: su columna ON == OFF "
                   "siempre. El LLM compensa la miopía de greedy/beam (lift positivo).")

        # --- Vista por caso (cobertura LLM ON) ---
        st.markdown("#### Cobertura por caso (LLM ON)")
        per = []
        for cid in case_ids:
            sample = next(r for r in rows if r["id"] == cid)
            row = {"ID": cid, "Factib.": FEAS_ICON[sample["feasibility"]],
                   "Objetivo": sample["target"]}
            for algo in BENCH_ALGOS:
                rr = _bench_get(rows, cid, algo, "on")
                row[ALGO_NAMES[algo]] = f"{rr['coverage']:g}%"
            per.append(row)
        st.dataframe(per, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        # --- RQ2: por factibilidad (ON) ---
        with c1:
            st.markdown("#### Por factibilidad (RQ2, ON)")
            feas_rows = []
            for feas in ["factible", "infactible_presupuesto", "infactible_catalogo"]:
                sub = [r for r in rows if r["feasibility"] == feas and r["condition"] == "on"]
                if not sub:
                    continue
                nc = len({r["id"] for r in sub})
                d = {"Factibilidad": FEAS_ICON[feas], "casos": nc}
                for algo in BENCH_ALGOS:
                    rs = [r for r in sub if r["algorithm"] == algo]
                    d[ALGO_NAMES[algo]] = f"{_avg(rs, 'coverage')}%"
                feas_rows.append(d)
            st.dataframe(feas_rows, hide_index=True, use_container_width=True)
        # --- RQ3: gap de optimalidad (ON) ---
        with c2:
            st.markdown("#### Gap de optimalidad (RQ3, ON)")
            gap_rows = []
            for algo in BENCH_ALGOS:
                rs = [r for r in rows if r["algorithm"] == algo
                      and r["condition"] == "on" and r["gap"] is not None]
                if not rs:
                    gap_rows.append({"Algoritmo": ALGO_NAMES[algo], "n": 0,
                                     "gap medio": "—", "gap máx": "—", "óptimos": "—"})
                    continue
                gaps = [r["gap"] for r in rs]
                opt = sum(1 for g in gaps if g == 0)
                gap_rows.append({
                    "Algoritmo": ALGO_NAMES[algo], "n": len(rs),
                    "gap medio": f"+{sum(gaps) / len(gaps):.1f}h",
                    "gap máx": f"+{max(gaps):.0f}h", "óptimos": f"{opt}/{len(rs)}"})
            st.dataframe(gap_rows, hide_index=True, use_container_width=True)

        # --- Casos donde el LLM cambió la ruta ---
        with st.expander("Casos donde el LLM cambió la ruta (ON vs OFF)"):
            changed = []
            for cid in case_ids:
                for algo in BENCH_ALGOS:
                    on = _bench_get(rows, cid, algo, "on")
                    off = _bench_get(rows, cid, algo, "off")
                    if (on["coverage"], on["hours"], on["resources"]) != \
                       (off["coverage"], off["hours"], off["resources"]):
                        changed.append({
                            "ID": cid, "Algoritmo": ALGO_NAMES[algo],
                            "ON": f"{on['coverage']:g}% · {on['hours']:g}h · {on['resources']}r",
                            "OFF": f"{off['coverage']:g}% · {off['hours']:g}h · {off['resources']}r"})
            if changed:
                st.dataframe(changed, hide_index=True, use_container_width=True)
            else:
                st.caption("Ninguna ruta cambió con el snapshot actual.")

        # --- Detalle de ruta por caso ---
        with st.expander("Ver la ruta generada de un caso"):
            dcid = st.selectbox("Caso", case_ids,
                                format_func=lambda c: f"{c} — {next(r for r in rows if r['id']==c)['profile']}",
                                key="bench_detail")
            dcond = st.radio("Condición", ["on", "off"], horizontal=True,
                             format_func=lambda c: "LLM ON" if c == "on" else "LLM OFF",
                             key="bench_cond")
            for algo in BENCH_ALGOS:
                r = _bench_get(rows, dcid, algo, dcond)
                st.markdown(f"**{ALGO_NAMES[algo]}** — {r['coverage']:g}% · "
                            f"{r['hours']:g}h · {len(r['path'])} recursos")
                if r["path"]:
                    pasos = " → ".join(graph.get_resource(rid)["name"] for rid in r["path"])
                    st.caption(pasos)
                else:
                    st.caption("_(ruta vacía)_")


# Al pulsar generar, CONGELAR los inputs actuales en una instantánea. La ruta solo
# se (re)genera con el click: Streamlit re-ejecuta el script en cada cambio de widget,
# pero render_results lee de esta instantánea, no de los widgets en vivo, así que
# editar el texto o los controles después NO dispara una ruta nueva.
if go_nl:
    st.session_state["request"] = {
        "mode": "nl",
        "nl_input": st.session_state.get("nl_input", ""),
        "sel_target": [], "sel_known": [], "sel_hours": None,
    }
elif go_ctrl:
    st.session_state["request"] = {
        "mode": "ctrl",
        "nl_input": "",
        "sel_target": list(st.session_state.get("sel_target", [])),
        "sel_known": list(st.session_state.get("sel_known", [])),
        "sel_hours": st.session_state.get("sel_hours", None),
    }


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------
def render_results(req: dict) -> None:
    # Lee de la instantánea congelada al pulsar generar, NO de los widgets en vivo.
    goal = resolve_goal(
        req["mode"],
        req.get("nl_input", ""),
        req.get("sel_target", []),
        req.get("sel_known", []),
        req.get("sel_hours", None),
    )

    # --- Fuera de alcance / sin objetivo mapeable ---
    if goal is None or not goal["target"]:
        st.warning("**Fuera de alcance.** " + domain.out_of_scope_message(graph))
        return

    target = set(goal["target"])
    known = set(goal["known"])
    max_hours = goal["max_hours"]

    # Objetivo interpretado
    st.markdown(ui.section_header("Tu objetivo, interpretado", "🎯"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Metas**\n\n{', '.join(sorted(target))}")
    c2.markdown(f"**Ya sabes**\n\n{', '.join(sorted(known)) or '_nada aún_'}")
    c3.markdown(f"**Horas**\n\n{max_hours if max_hours is not None else 'sin límite'}")

    # --- Caso borde: ya sabes todo lo que pides (target ⊆ known) ---
    # No hay nada nuevo que aprender: cortar antes de optimizar/llamar al LLM.
    if not (target - known):
        st.write("")
        st.markdown(ui.already_known_card(sorted(target)), unsafe_allow_html=True)
        return

    # --- Factibilidad ---
    feas = PathOptimizer(graph, {}).check_feasibility(target, known, max_hours)
    feas_label = classify(feas)
    if feas["unreachable_skills"]:
        st.markdown(
            ui.feasibility_badge("infactible_catalogo",
                                 "Algunas metas no están en el catálogo."),
            unsafe_allow_html=True)
        st.info("**Fuera del catálogo:** " + ", ".join(feas["unreachable_skills"])
                + ". Se planifica solo lo que sí cubrimos. "
                + domain.out_of_scope_message(graph))
        target = set(feas["reachable_skills"])
        if not target:
            return
        feas = PathOptimizer(graph, {}).check_feasibility(target, known, max_hours)
        feas_label = classify(feas)

    st.markdown(ui.feasibility_badge(feas_label, feas["message"]),
                unsafe_allow_html=True)
    st.write("")

    # --- Señal del LLM (scores) ---
    llm_scores = {}
    if use_llm:
        with st.spinner("Evaluando relevancia de recursos con IA…"):
            llm_scores = llm.score_resources_for_goal(goal["goal_summary"],
                                                      list(graph.resources.values()))

    # --- Tres rutas ---
    optimizer = PathOptimizer(graph, llm_scores=llm_scores)
    comparison = optimizer.compare(target, known, max_hours)
    min_hours = feas["min_hours_needed"]

    # Recomendación del LLM (con fallback interno)
    with st.spinner("Comparando algoritmos…"):
        analysis = llm.compare_algorithms(
            comparison["greedy"], comparison["beam_search"],
            comparison["a_star"], goal["goal_summary"])
    reco_key = analysis.get("recommended", "a_star")
    if reco_key not in comparison:
        reco_key = "a_star"

    st.markdown(ui.section_header("Tres estrategias, una comparación", "🧮"),
                unsafe_allow_html=True)
    st.caption(f"Señal del LLM: **{'ACTIVADA' if use_llm else 'desactivada'}** · "
               f"el gap se mide solo en objetivos factibles cubiertos al 100%.")
    cols = st.columns(3)
    for col, algo in zip(cols, ["greedy", "beam_search", "a_star"]):
        r = comparison[algo]
        gap = (r["total_hours"] - min_hours) if (
            r["coverage_pct"] == 100.0 and feas_label == "factible") else None
        col.markdown(ui.route_card(r, recommended=(algo == reco_key), gap=gap),
                     unsafe_allow_html=True)

    # --- Recomendación + explicación ---
    st.write("")
    st.markdown(ui.section_header("Recomendación de la IA", "💡"), unsafe_allow_html=True)
    reco = comparison[reco_key]
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        meta = ui.ALGO_META[reco_key]
        st.markdown(f"### {meta['icon']} {meta['name']}")
        st.caption(meta["tag"])
        st.metric("Cobertura", f"{reco['coverage_pct']:g}%")
        st.metric("Horas", f"{reco['total_hours']:g}h")
    with rc2:
        st.markdown(f"**Por qué:** {analysis.get('reason', '—')}")
        st.markdown(f"**Trade-off:** {analysis.get('tradeoff', '—')}")
        with st.spinner("Redactando tu explicación personalizada…"):
            explanation = llm.explain_path(reco, goal["goal_summary"],
                                           graph.get_resource)
        st.info(explanation)

    # --- Timeline ---
    st.write("")
    st.markdown(ui.section_header("Tu ruta, paso a paso", "🗺️"), unsafe_allow_html=True)
    st.markdown(ui.route_timeline(reco, graph.get_resource), unsafe_allow_html=True)
    if reco["skills_missing"]:
        st.caption("No cubierto en esta ruta: " + ", ".join(reco["skills_missing"]))

    # --- Export ---
    md = ui.plan_to_markdown(reco, graph.get_resource, goal=goal["goal_summary"],
                             algorithm_name=ui.ALGO_META[reco_key]["name"],
                             feasibility_msg=feas["message"])
    st.download_button("⬇️  Exportar plan (Markdown)", md,
                       file_name="mi_ruta_de_aprendizaje.md", mime="text/markdown")

    # --- Modo ablación: ON vs OFF en vivo (solo optimizador, sin nuevas llamadas) ---
    st.write("")
    with st.expander("🔬  Modo ablación — efecto del LLM (ON vs OFF) en vivo"):
        st.caption("Mismo objetivo, dos condiciones. A* no cambia (independiente del "
                   "LLM por construcción); greedy/beam sí: el prior del LLM compensa "
                   "su miopía en cadenas profundas, pero puede dañar en objetivos "
                   "dispersos. Este es el resultado central del proyecto.")
        if not llm_scores:
            scores_for_demo = llm.score_resources_for_goal(
                goal["goal_summary"], list(graph.resources.values()))
        else:
            scores_for_demo = llm_scores
        comp_on = PathOptimizer(graph, scores_for_demo).compare(target, known, max_hours)
        comp_off = PathOptimizer(graph, {}).compare(target, known, max_hours)
        ac1, ac2 = st.columns(2)
        ac1.markdown("**🟢 LLM ON**")
        ac2.markdown("**⚪ LLM OFF**")
        for algo in ["greedy", "beam_search", "a_star"]:
            on_c = comp_on[algo]["coverage_pct"]
            off_c = comp_off[algo]["coverage_pct"]
            delta = on_c - off_c
            sign = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            nm = ui.ALGO_META[algo]["name"]
            ac1.markdown(f"{nm}: **{on_c:g}%** · {on_c and comp_on[algo]['total_hours']:g}h")
            ac2.markdown(f"{nm}: **{off_c:g}%** · {sign} {abs(delta):g} pts")


# Renderizar la ruta DENTRO del contenedor de la pestaña donde se generó (NL o
# controles), no debajo de todas las pestañas. Así, en la pestaña de tests no
# aparece ninguna ruta vieja al hacer scroll.
if "request" in st.session_state:
    _req = st.session_state["request"]
    with (nl_results if _req["mode"] == "nl" else ctrl_results):
        st.write("")
        render_results(_req)
else:
    with nl_results:
        st.write("")
        st.markdown('<p class="muted">⬆️ Describe tu objetivo o usa los controles, y '
                    'pulsa <b>Generar mi ruta</b>.</p>', unsafe_allow_html=True)
