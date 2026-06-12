"""
Capa de presentación para la UI Streamlit (Fase 6).

Builders puros de HTML/CSS y Markdown: NO importan streamlit ni la lógica de
negocio. ``app.py`` los orquesta. Mantener aquí todo el "look" (paleta azul
profundo + azul cielo, tarjetas glassmorphism, badges) deja la app legible y
estos helpers testeables de forma aislada.

Paleta:
  - Azul profundo (fondo)   #0a1929 / #0f2744
  - Azul cielo (acentos)    #38bdf8
  - Azul (gradiente)        #2563eb
  - Factible / ajustado / inalcanzable  verde / ámbar / rojo
"""

# ----------------------------------------------------------------------
# Paleta y metadatos de algoritmos
# ----------------------------------------------------------------------
SKY = "#38bdf8"
BLUE = "#2563eb"
DEEP = "#0a1929"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"

ALGO_META: dict[str, dict[str, str]] = {
    "greedy": {"name": "Greedy", "tag": "Voraz + lookahead", "icon": "⚡"},
    "beam_search": {"name": "Beam Search", "tag": "Haz de búsqueda", "icon": "🔦"},
    "a_star": {"name": "A*", "tag": "Óptimo en horas", "icon": "⭐"},
}

FEAS_META: dict[str, dict[str, str]] = {
    "factible": {"label": "Factible", "color": GREEN, "icon": "✓"},
    "infactible_presupuesto": {"label": "Presupuesto ajustado", "color": AMBER, "icon": "◐"},
    "infactible_catalogo": {"label": "Fuera del catálogo", "color": RED, "icon": "✕"},
}


# ----------------------------------------------------------------------
# Ejemplos de prompts (perfiles reales de tests/test_cases.py, en lenguaje natural)
# ----------------------------------------------------------------------
EXAMPLE_PROMPTS: list[dict[str, str]] = [
    {"icon": "🌐", "title": "Full-stack desde cero",
     "text": "Quiero ser desarrollador web full-stack desde cero: aprender React, "
             "backend y testing. Tengo unas 80 horas."},
    {"icon": "🤖", "title": "LLMs y RAG",
     "text": "Soy ML engineer y quiero especializarme en LLMs, fine-tuning y "
             "sistemas RAG. Ya sé Python y machine learning. Dispongo de 110 horas."},
    {"icon": "🛠️", "title": "Salto a DevOps",
     "text": "Quiero dar el salto a DevOps: Kubernetes, CI/CD y Terraform. Sé git "
             "y tengo unas 90 horas."},
    {"icon": "📊", "title": "Ingeniero de datos",
     "text": "Soy analista y quiero convertirme en ingeniero de datos: Spark, data "
             "warehouse y Airflow. Sé Python y SQL. Tengo 45 horas."},
    {"icon": "🧠", "title": "Entrar a ML",
     "text": "Soy principiante y quiero entrar a machine learning: modelos "
             "supervisados, sklearn y evaluación. No sé nada aún, tengo 80 horas."},
]


# ----------------------------------------------------------------------
# CSS global
# ----------------------------------------------------------------------
def css() -> str:
    """Bloque <style> con el tema pro. Se inyecta una vez con st.markdown."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {{
  --sky: {SKY};
  --blue: {BLUE};
  --muted: {MUTED};
}}

/* Fondo con resplandor radial azul */
.stApp {{
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(1000px 500px at 110% 10%, rgba(37,99,235,0.12), transparent 55%),
    linear-gradient(180deg, #081320 0%, {DEEP} 40%, #07111d 100%);
}}

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.5px; }}

/* Ocultar chrome por defecto, PERO conservar el control para reabrir el sidebar.
   El botón de expandir (stExpandSidebarButton) vive DENTRO del toolbar que
   ocultamos; visibility se revierte en el hijo para mostrar solo ese botón. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{
  visibility: visible !important; z-index: 1000;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}

/* ---------- Hero ---------- */
.hero {{
  border-radius: 22px;
  padding: 30px 34px;
  background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(37,99,235,0.10));
  border: 1px solid rgba(56,189,248,0.22);
  box-shadow: 0 20px 60px -25px rgba(56,189,248,0.5);
  position: relative; overflow: hidden;
}}
.hero::after {{
  content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
  background: radial-gradient(circle, rgba(56,189,248,0.35), transparent 70%);
}}
.hero h1 {{
  margin: 0; font-size: 2.35rem; line-height: 1.1;
  background: linear-gradient(120deg, #e0f2fe 10%, {SKY} 50%, {BLUE} 95%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.hero p {{ margin: 10px 0 0; color: {MUTED}; font-size: 1.02rem; max-width: 720px; }}
.hero .pill {{
  display:inline-block; margin-top:16px; padding:5px 14px; border-radius:999px;
  font-size:0.78rem; font-weight:600; letter-spacing:0.3px;
  color:{SKY}; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.30);
}}

/* ---------- Tarjetas ---------- */
.card {{
  border-radius: 18px; padding: 20px 22px; height: 100%;
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(148,163,184,0.16);
  backdrop-filter: blur(6px);
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
.card:hover {{ transform: translateY(-3px); border-color: rgba(56,189,248,0.4); }}
.card.reco {{
  border: 1px solid rgba(56,189,248,0.55);
  box-shadow: 0 18px 50px -22px rgba(56,189,248,0.65);
  background: linear-gradient(160deg, rgba(56,189,248,0.10), rgba(255,255,255,0.03));
}}
.card .algo {{ display:flex; align-items:center; gap:10px; }}
.card .algo .nm {{ font-family:'Space Grotesk'; font-size:1.25rem; font-weight:700; color:#f1f5f9; }}
.card .algo .tg {{ font-size:0.72rem; color:{MUTED}; }}
.reco-flag {{
  float:right; font-size:0.68rem; font-weight:700; letter-spacing:0.4px;
  color:{DEEP}; background:linear-gradient(120deg,{SKY},{BLUE});
  padding:3px 10px; border-radius:999px;
}}

/* Métricas dentro de tarjeta */
.metrics {{ display:flex; gap:14px; margin:16px 0 6px; }}
.metric {{ flex:1; }}
.metric .val {{ font-family:'Space Grotesk'; font-size:1.5rem; font-weight:700; color:{SKY}; }}
.metric .lbl {{ font-size:0.68rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.5px; }}

/* Barra de cobertura */
.cov-track {{ height:8px; border-radius:999px; background:rgba(148,163,184,0.18); overflow:hidden; margin-top:4px; }}
.cov-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg,{SKY},{BLUE}); }}

/* ---------- Badge factibilidad ---------- */
.fbadge {{
  display:inline-flex; align-items:center; gap:9px; padding:10px 18px; border-radius:14px;
  font-weight:600; font-size:0.95rem;
}}
.fbadge .dot {{ width:10px; height:10px; border-radius:50%; box-shadow:0 0 12px currentColor; }}

/* ---------- Timeline ---------- */
.tl-step {{
  display:flex; gap:14px; padding:12px 0; border-left:2px solid rgba(56,189,248,0.30);
  margin-left:10px; padding-left:20px; position:relative;
}}
.tl-step::before {{
  content:""; position:absolute; left:-7px; top:18px; width:12px; height:12px; border-radius:50%;
  background:{SKY}; box-shadow:0 0 0 4px rgba(56,189,248,0.18);
}}
.tl-step .ix {{ color:{SKY}; font-weight:700; font-family:'Space Grotesk'; min-width:24px; }}
.tl-step .body .t {{ font-weight:600; color:#f1f5f9; }}
.tl-step .body .m {{ font-size:0.8rem; color:{MUTED}; margin-top:2px; }}
.tl-step .body .sk {{ font-size:0.74rem; color:{SKY}; margin-top:4px; }}

/* ---------- Chips de dominio (catálogo) ---------- */
.dchip {{
  display:inline-block; margin:3px 4px; padding:3px 10px; border-radius:999px;
  font-size:0.72rem; color:#cbd5e1; background:rgba(56,189,248,0.08);
  border:1px solid rgba(56,189,248,0.18);
}}
.dom-title {{ color:{SKY}; font-weight:700; font-size:0.9rem; margin:10px 0 4px; font-family:'Space Grotesk'; }}

/* Botones */
.stButton > button {{
  border-radius: 12px; font-weight: 600; border:1px solid rgba(56,189,248,0.30);
  background: rgba(56,189,248,0.08); color:#e0f2fe; transition: all .15s ease;
}}
.stButton > button:hover {{ border-color:{SKY}; background:rgba(56,189,248,0.16); color:#fff; }}
[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(120deg, {SKY}, {BLUE}) !important; color:#04111f !important;
  border:none !important; box-shadow:0 10px 30px -10px rgba(56,189,248,0.7) !important;
}}

.section-h {{ font-family:'Space Grotesk'; font-weight:700; color:#e2e8f0; font-size:1.25rem;
  margin: 8px 0 2px; display:flex; align-items:center; gap:9px; }}
.section-h .bar {{ width:4px; height:20px; border-radius:4px; background:linear-gradient(180deg,{SKY},{BLUE}); }}
.muted {{ color:{MUTED}; font-size:0.9rem; }}
</style>
"""


# ----------------------------------------------------------------------
# Builders de HTML
# ----------------------------------------------------------------------
def hero(domain_name: str, n_resources: int, n_skills: int, n_domains: int) -> str:
    return f"""
<div class="hero">
  <h1>Planificador de Rutas de Aprendizaje · IA</h1>
  <p>Construye una secuencia óptima de recursos para tu objetivo, respetando
     prerequisitos y tu presupuesto de horas. Tres algoritmos de búsqueda guiados
     por un LLM.</p>
  <span class="pill">{domain_name} &nbsp;·&nbsp; {n_resources} recursos &nbsp;·&nbsp;
     {n_skills} habilidades &nbsp;·&nbsp; {n_domains} áreas</span>
</div>
"""


def section_header(title: str, icon: str = "") -> str:
    return f'<div class="section-h"><span class="bar"></span>{icon} {title}</div>'


def already_known_card(skills: list[str]) -> str:
    """Caso borde: el objetivo ya está cubierto por lo que el usuario sabe
    (target ⊆ known). No hay ruta que construir; se felicita y se sugiere ampliar."""
    chips = "".join(f'<span class="dchip">{s}</span>' for s in skills)
    return f"""
<div class="card" style="border:1px solid {GREEN}55;
     background:linear-gradient(160deg, {GREEN}14, rgba(255,255,255,0.03));">
  <div style="font-family:'Space Grotesk'; font-size:1.35rem; font-weight:700; color:{GREEN};">
     🎉 ¡Ya dominas lo que pediste!
  </div>
  <p style="color:{TEXT}; margin:10px 0 6px;">
     Las habilidades de tu objetivo ya están entre las que dices conocer, así que
     <b>no hace falta construir una ruta</b>: no hay nada nuevo que aprender aquí.
  </p>
  <div style="margin:6px 0 10px;">{chips}</div>
  <p class="muted" style="margin:0;">¿Quieres ir más allá? Añade habilidades
     <i>nuevas</i> a tu objetivo (p. ej. un nivel más avanzado o un área distinta)
     y vuelve a generar la ruta.</p>
</div>
"""


def feasibility_badge(feas_label: str, message: str) -> str:
    meta = FEAS_META.get(feas_label, FEAS_META["infactible_catalogo"])
    c = meta["color"]
    return f"""
<div class="fbadge" style="color:{c}; background:{c}1a; border:1px solid {c}55;">
  <span class="dot" style="background:{c};"></span>
  <span>{meta['icon']} {meta['label']}</span>
  <span style="color:{MUTED}; font-weight:400; font-size:0.85rem;">— {message}</span>
</div>
"""


def _metric(val: str, lbl: str) -> str:
    return f'<div class="metric"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def route_card(result: dict, *, recommended: bool = False,
               gap: float | None = None) -> str:
    """Tarjeta de una ruta (greedy/beam/a_star) con métricas y barra de cobertura."""
    meta = ALGO_META.get(result["algorithm"], {"name": result["algorithm"], "tag": "", "icon": "•"})
    cov = result["coverage_pct"]
    flag = '<span class="reco-flag">★ RECOMENDADA</span>' if recommended else ""
    gap_metric = ""
    if gap is not None:
        gap_txt = "óptimo" if gap == 0 else f"+{gap:g}h"
        gap_metric = _metric(gap_txt, "vs óptimo")
    return f"""
<div class="card {'reco' if recommended else ''}">
  {flag}
  <div class="algo">
    <span style="font-size:1.5rem;">{meta['icon']}</span>
    <div><div class="nm">{meta['name']}</div><div class="tg">{meta['tag']}</div></div>
  </div>
  <div class="metrics">
    {_metric(f"{result['total_hours']:g}h", "horas")}
    {_metric(f"{len(result['path'])}", "recursos")}
    {gap_metric}
  </div>
  <div style="display:flex; justify-content:space-between; align-items:baseline;">
    <span class="lbl" style="color:{MUTED}; font-size:0.7rem; text-transform:uppercase;">cobertura</span>
    <span style="color:{SKY}; font-weight:700; font-family:'Space Grotesk';">{cov:g}%</span>
  </div>
  <div class="cov-track"><div class="cov-fill" style="width:{cov}%;"></div></div>
</div>
"""


def route_timeline(result: dict, get_resource) -> str:
    """Línea de tiempo de la ruta recomendada: cada paso con sus prerequisitos."""
    if not result["path"]:
        return '<p class="muted">No hay recursos en la ruta.</p>'
    steps = []
    acquired: set[str] = set()
    for i, rid in enumerate(result["path"], 1):
        r = get_resource(rid)
        reqs = r.get("requires", [])
        req_txt = (
            f'<div class="m">requiere: {", ".join(reqs)}</div>' if reqs
            else '<div class="m">sin prerequisitos — punto de entrada</div>'
        )
        steps.append(f"""
<div class="tl-step">
  <span class="ix">{i}</span>
  <div class="body">
    <div class="t">{r['name']} <span style="color:{MUTED};font-weight:400;">· {r['duration_hours']:g}h · {r['type']}</span></div>
    {req_txt}
    <div class="sk">enseña: {", ".join(r['teaches'])}</div>
  </div>
</div>""")
        acquired.update(r["teaches"])
    return "".join(steps)


def domain_catalog(skills_by_domain: dict[str, list[str]], labels: dict[str, str]) -> str:
    """Explorador del catálogo: skills agrupadas por dominio (para el sidebar)."""
    blocks = []
    for dom in sorted(skills_by_domain):
        label = labels.get(dom, dom)
        chips = "".join(f'<span class="dchip">{s}</span>' for s in skills_by_domain[dom])
        blocks.append(
            f'<div class="dom-title">{label} '
            f'<span style="color:{MUTED};font-weight:400;">({len(skills_by_domain[dom])})</span></div>'
            f'<div>{chips}</div>'
        )
    return "".join(blocks)


# ----------------------------------------------------------------------
# Export a Markdown
# ----------------------------------------------------------------------
def plan_to_markdown(result: dict, get_resource, *, goal: str, algorithm_name: str,
                     feasibility_msg: str) -> str:
    """Genera el plan recomendado como Markdown descargable (checklist)."""
    lines = [
        "# Mi ruta de aprendizaje",
        "",
        f"**Objetivo:** {goal}",
        f"**Algoritmo:** {algorithm_name}",
        f"**Cobertura:** {result['coverage_pct']:g}%  ·  **Horas totales:** {result['total_hours']:g}h"
        f"  ·  **Recursos:** {len(result['path'])}",
        f"**Factibilidad:** {feasibility_msg}",
        "",
        "## Secuencia (en orden)",
        "",
    ]
    acc_h = 0.0
    for i, rid in enumerate(result["path"], 1):
        r = get_resource(rid)
        acc_h += r["duration_hours"]
        reqs = f" _(requiere: {', '.join(r['requires'])})_" if r.get("requires") else ""
        lines.append(
            f"- [ ] **{i}. {r['name']}** — {r['duration_hours']:g}h · {r['type']} "
            f"· acumulado {acc_h:g}h{reqs}  \n"
            f"      enseña: {', '.join(r['teaches'])}"
        )
    if result["skills_missing"]:
        lines += ["", "## Habilidades no cubiertas", "",
                  ", ".join(result["skills_missing"])]
    lines += ["", "---", "_Generado por el Planificador de Rutas de Aprendizaje · IA_"]
    return "\n".join(lines)
