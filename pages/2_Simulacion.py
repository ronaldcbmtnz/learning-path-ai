"""
Pagina Streamlit del modulo de Simulacion (multipagina nativa).

AISLAMIENTO DE ERRORES (HANDOFF_SIMULACION.md §7): todo el backend de simulacion se
importa de forma PEREZOSA y el cuerpo va envuelto en try/except. Si algo de
simulacion falla, esta pagina muestra el error pero la pagina principal (app.py, el
modulo de IA) sigue intacta. Streamlit detecta pages/ automaticamente: app.py NO se
modifica.

El tema oscuro se reutiliza de src/ui.py (import de solo lectura; no se modifica).
"""
import streamlit as st

st.set_page_config(page_title="Simulacion - Rutas IA", page_icon="🎲", layout="wide")

# Tema (reutiliza el CSS del modulo de IA; ui.py es de solo lectura aqui).
try:
    from src import ui as _ui
    st.markdown(_ui.css(), unsafe_allow_html=True)
except Exception:
    _ui = None


def _header(title: str, icon: str = "") -> None:
    if _ui is not None:
        st.markdown(_ui.section_header(title, icon), unsafe_allow_html=True)
    else:
        st.subheader(f"{icon} {title}")


st.markdown(
    "<h1 style='font-family:Space Grotesk,sans-serif;"
    "background:linear-gradient(120deg,#e0f2fe,#38bdf8,#2563eb);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
    "🎲 Modulo de Simulacion</h1>",
    unsafe_allow_html=True,
)
st.caption("Logica difusa (Cap. 5) + Monte Carlo del ruido del LLM (Caps. 1, 2, 4). "
           "Aislado del modulo de IA: si esto falla, el planificador sigue funcionando.")

# --- Import perezoso del backend de simulacion, con aislamiento de errores ---
try:
    import json
    from src.graph import ResourceGraph
    from src import fuzzy_scorer
    from src import simulation
    from src import sim_ui
    from tests.test_cases import TEST_CASES
    _SIM_OK = True
    _SIM_ERR = None
except Exception as exc:  # pragma: no cover - red de aislamiento
    _SIM_OK = False
    _SIM_ERR = exc

if not _SIM_OK:
    st.error(f"No se pudo cargar el modulo de simulacion: {_SIM_ERR}")
    st.info("La pagina principal (planificador de IA) no se ve afectada por esto.")
    st.stop()


@st.cache_resource
def _get_graph():
    return ResourceGraph()


@st.cache_data
def _get_snapshot():
    with open("data/llm_scores_snapshot.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Corriendo Monte Carlo...")
def _run_sim(tc_id: str, sigmas: tuple, n_runs: int, seed: int) -> dict:
    graph = _get_graph()
    snap = _get_snapshot()
    tc = next(t for t in TEST_CASES if t["id"] == tc_id)
    return simulation.simulate(
        graph, snap[tc_id]["scores"], tc["target_skills"], tc["known_skills"],
        tc["max_hours"], list(sigmas), n_runs=n_runs, seed=seed,
    )


try:
    graph = _get_graph()
    snapshot = _get_snapshot()
    case_ids = sorted(snapshot.keys())
    tc_by_id = {t["id"]: t for t in TEST_CASES}

    def _case_label(cid: str) -> str:
        """Etiqueta explicativa de un caso: que quiere aprender, que ya sabe y horas
        (no solo el perfil, que por si solo no dice las habilidades objetivo)."""
        tc = tc_by_id[cid]
        objetivo = ", ".join(sorted(tc["target_skills"]))
        previas = ", ".join(sorted(tc["known_skills"])) or "nada"
        horas = tc["max_hours"] if tc["max_hours"] is not None else "sin limite"
        return (f"{cid} · {tc['profile']}  |  aprender: {objetivo}  |  "
                f"ya sabe: {previas}  |  {horas} h")

    tab_fuzzy, tab_ablation, tab_mc = st.tabs(
        ["🔢 Sistema difuso", "⚖️ Lineal vs Difuso", "🎲 Monte Carlo"]
    )

    # ==================================================================
    # TAB 1 - El sistema difuso, visible
    # ==================================================================
    with tab_fuzzy:
        _header("Funciones de membresia", "🔢")
        st.caption("Cada variable linguistica se describe con triangulos. "
                   f"Enumerar todas las reglas daria {fuzzy_scorer.FULL_RULE_COUNT}; "
                   f"la base reducida usa {fuzzy_scorer.REDUCED_RULE_COUNT} "
                   "(hallazgo 5.2.2: explosion de reglas).")
        st.pyplot(sim_ui.fig_memberships(), use_container_width=True)
        st.pyplot(sim_ui.fig_output_membership(), use_container_width=True)

        st.divider()
        _header("Inferencia paso a paso", "🔎")
        st.caption("Mueve las entradas y observa fuzzificacion -> reglas -> centroide.")
        c1, c2, c3, c4 = st.columns(4)
        relev = c1.slider("relevancia (LLM)", 0.0, 1.0, 0.8, 0.05)
        cob = c2.slider("cobertura", 0.0, 1.0, 0.7, 0.05)
        hrs = c3.slider("horas", 0.0, 1.0, 0.2, 0.05)
        salto = c4.slider("salto dificultad", 0.0, 1.0, 0.0, 0.05)
        features = {"relevancia": relev, "cobertura": cob, "horas": hrs, "salto": salto}
        utility, trace = fuzzy_scorer.fuzzy_utility_trace(features)

        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Utilidad (centroide)", f"{utility:.1f} / 100")
            st.markdown("**Reglas activadas**")
            if trace["fired_rules"]:
                for fr in trace["fired_rules"][:8]:
                    ant = ", ".join(f"{k}={v}" for k, v in fr["antecedents"].items())
                    st.markdown(f"- `{fr['strength']:.2f}` · SI {ant} -> **{fr['consequent']}**")
            else:
                st.markdown("_ninguna regla activa -> utilidad neutra (50)_")
        with m2:
            st.pyplot(sim_ui.fig_inference_trace(trace, utility), use_container_width=True)

    # ==================================================================
    # TAB 2 - Lineal vs Difuso (ablacion de la politica, sin ruido)
    # ==================================================================
    with tab_ablation:
        _header("Politica de decision: lineal vs difusa", "⚖️")
        st.caption("Misma señal del LLM (snapshot, sin ruido); cambia SOLO como se "
                   "combinan los factores. A* es identico en ambas (no usa el scoring).")
        cid = st.selectbox("Caso (perfil)", case_ids,
                            format_func=_case_label, key="abl_case")
        tc = tc_by_id[cid]
        scores = snapshot[cid]["scores"]
        from src.optimizer import PathOptimizer
        from src.sim_optimizer import FuzzyPathOptimizer
        bw = max(3, min(6, len(tc["target_skills"])))
        cols = st.columns(2)
        for col, (policy, opt) in zip(cols, [
            ("Lineal", PathOptimizer(graph, scores)),
            ("Difusa", FuzzyPathOptimizer(graph, scores)),
        ]):
            with col:
                st.markdown(f"### Politica {policy}")
                for algo in ("greedy", "beam_search", "astar"):
                    if algo == "beam_search":
                        r = opt.beam_search(tc["target_skills"], tc["known_skills"], tc["max_hours"], beam_width=bw)
                    else:
                        r = getattr(opt, algo)(tc["target_skills"], tc["known_skills"], tc["max_hours"])
                    st.markdown(
                        f"**{algo}** — {r['coverage_pct']}% · {r['total_hours']}h · "
                        f"{len(r['path'])} recursos")

    # ==================================================================
    # TAB 3 - Monte Carlo del ruido del LLM
    # ==================================================================
    with tab_mc:
        _header("Monte Carlo: robustez al ruido del LLM", "🎲")
        st.caption("El LLM 'cambia de opinion' cada corrida. Se perturba cada score "
                   "con ruido triangular en [-sigma, sigma] y se mide la dispersion.")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cid = cc1.selectbox("Caso", case_ids,
                            format_func=_case_label, key="mc_case")
        n_runs = cc2.slider("Corridas (N)", 20, 300, 100, 20)
        seed = cc3.number_input("Semilla", 0, 9999, 0, 1)
        sigma_max = cc4.slider("sigma max", 0.1, 0.5, 0.4, 0.1)
        sigmas = tuple(round(0.1 * k, 1) for k in range(0, int(sigma_max * 10) + 1))

        if st.button("▶ Correr simulacion", type="primary"):
            st.session_state["mc_result"] = _run_sim(cid, sigmas, n_runs, int(seed))
            st.session_state["mc_case_run"] = cid

        result = st.session_state.get("mc_result")
        if result:
            st.markdown(f"**Objetivo:** {tc_by_id[st.session_state['mc_case_run']]['profile']}  ·  "
                        f"A* (control) = {result['astar_cov']:.0f}% en {result['astar_hours']}h")
            cg, cb = st.columns(2)
            with cg:
                st.pyplot(sim_ui.fig_robustness(result, "greedy"), use_container_width=True)
            with cb:
                st.pyplot(sim_ui.fig_robustness(result, "beam_search"), use_container_width=True)

            st.divider()
            _header("Distribuciones a un sigma dado", "📊")
            sig_sel = st.select_slider("sigma", options=list(result["sigmas"]),
                                       value=result["sigmas"][-1])
            hg, hb = st.columns(2)
            with hg:
                st.pyplot(sim_ui.fig_histograms(result, sig_sel, "greedy"), use_container_width=True)
            with hb:
                st.pyplot(sim_ui.fig_histograms(result, sig_sel, "beam_search"), use_container_width=True)

            st.divider()
            _header("Media +- varianza (cobertura %)", "📋")
            rows = []
            for r in result["records"]:
                rows.append({
                    "sigma": r["sigma"], "politica": r["policy"], "algoritmo": r["algorithm"],
                    "cob. media": r["cov_mean"], "cob. var": r["cov_var"],
                    "horas media": r["hours_mean"], "horas var": r["hours_var"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Elige un caso y pulsa **Correr simulacion**.")

except Exception as exc:  # pragma: no cover - aislamiento de errores de la pagina
    st.error(f"Error en la pagina de simulacion: {exc}")
    st.info("Esto NO afecta al planificador de IA (pagina principal).")
