"""
UI Streamlit del Planificador de Rutas de Aprendizaje (Fase 6).

Capa de PRESENTACIÓN: orquesta el mismo backend que el CLI (`ResourceGraph`,
`PathOptimizer`, `LLMClient`, `domain`) sin duplicar lógica de negocio. Todo el
"look" vive en `src/ui.py`; la acotación de dominio en `src/domain.py`.

Ejecutar:  streamlit run app.py
"""
import streamlit as st

from src.graph import ResourceGraph
from src.optimizer import PathOptimizer
from src.llm_client import LLMClient
from src import domain, ui

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


tab_nl, tab_ctrl = st.tabs(["💬  Lenguaje natural", "🎛️  Controles precisos"])

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

with tab_ctrl:
    st.markdown(ui.section_header("Arma tu objetivo a mano", "🎛️"), unsafe_allow_html=True)
    st.caption("Sin depender del parseo del LLM: elige habilidades meta, lo que ya "
               "dominas y tus horas.")
    sel_target = st.multiselect("🎯 Habilidades que quieres aprender", ALL_SKILLS,
                                key="sel_target")
    sel_known = st.multiselect("✅ Habilidades que ya tienes", ALL_SKILLS,
                               key="sel_known")
    sel_hours = st.slider("⏱️ Horas disponibles", 0, 200, 60, step=5, key="sel_hours")
    go_ctrl = st.button("🚀  Generar mi ruta", type="primary",
                        use_container_width=True, key="go_ctrl")


# Registrar la petición al pulsar generar (la fuente de input según la pestaña)
if go_nl:
    st.session_state["request"] = {"mode": "nl"}
elif go_ctrl:
    st.session_state["request"] = {"mode": "ctrl"}


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------
def render_results(req_mode: str) -> None:
    goal = resolve_goal(
        req_mode,
        st.session_state.get("nl_input", ""),
        st.session_state.get("sel_target", []),
        st.session_state.get("sel_known", []),
        st.session_state.get("sel_hours", None),
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


if "request" in st.session_state:
    st.write("")
    render_results(st.session_state["request"]["mode"])
else:
    st.write("")
    st.markdown('<p class="muted">⬆️ Describe tu objetivo o usa los controles, y '
                'pulsa <b>Generar mi ruta</b>.</p>', unsafe_allow_html=True)
