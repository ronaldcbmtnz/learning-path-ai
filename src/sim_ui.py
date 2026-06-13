"""
Builders de figuras matplotlib (tema oscuro tech) para la pagina de Simulacion.

AISLAMIENTO (HANDOFF_SIMULACION.md §4): NO importa streamlit ni la logica de IA.
Importa matplotlib y las definiciones del sistema difuso (src/fuzzy_scorer, modulo
de simulacion) para graficarlas. Recibe datos ya calculados (resultados del Monte
Carlo) y devuelve figuras; la pagina decide como mostrarlas (st.pyplot).
"""
import matplotlib
matplotlib.use("Agg")  # backend sin ventana (Streamlit renderiza la figura)
import matplotlib.pyplot as plt

from src.fuzzy_scorer import INPUT_MFS, OUTPUT_MF, OUTPUT_RANGE, trimf

# Paleta (espejo de src/ui.py, sin importarla para no acoplar)
SKY = "#38bdf8"
BLUE = "#2563eb"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
GRID = "#1e3a5f"
TEXT = "#cbd5e1"

POLICY_COLOR = {"lineal": AMBER, "difuso": SKY}
ALGO_COLOR = {"greedy": AMBER, "beam_search": SKY, "astar": GREEN}


def _style(ax) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("#e2e8f0")
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.6)


def _fig(w=7.0, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0.0)
    return fig, ax


def _sample(a, b, c, n=200):
    xs = [i / n for i in range(n + 1)]
    return xs, [trimf(x, a, b, c) for x in xs]


# ----------------------------------------------------------------------
# 1. Funciones de membresia de TODAS las variables de entrada
# ----------------------------------------------------------------------
def fig_memberships():
    variables = list(INPUT_MFS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.0))
    fig.patch.set_alpha(0.0)
    palette = [SKY, AMBER, GREEN, RED]
    for ax, var in zip(axes.flat, variables):
        for i, (label, (a, b, c)) in enumerate(INPUT_MFS[var].items()):
            xs, ys = _sample(a, b, c)
            ax.plot(xs, ys, color=palette[i % len(palette)], linewidth=2, label=label)
            ax.fill_between(xs, ys, color=palette[i % len(palette)], alpha=0.10)
        ax.set_title(var, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, facecolor="none", labelcolor=TEXT, edgecolor=GRID)
        _style(ax)
    fig.tight_layout()
    return fig


def fig_output_membership():
    fig, ax = _fig(7.5, 2.6)
    lo, hi = OUTPUT_RANGE
    palette = [RED, AMBER, "#a3a3a3", SKY, GREEN]
    for i, (label, (a, b, c)) in enumerate(OUTPUT_MF.items()):
        n = 300
        xs = [lo + (hi - lo) * k / n for k in range(n + 1)]
        ys = [trimf(x, a, b, c) for x in xs]
        ax.plot(xs, ys, color=palette[i % len(palette)], linewidth=2, label=label)
        ax.fill_between(xs, ys, color=palette[i % len(palette)], alpha=0.10)
    ax.set_title("utilidad (salida)", fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=5, facecolor="none", labelcolor=TEXT, edgecolor=GRID)
    _style(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Trace de inferencia para un recurso (fuzzificacion + salida agregada)
# ----------------------------------------------------------------------
def fig_inference_trace(trace: dict, utility: float):
    fig, ax = _fig(7.5, 3.0)
    lo, hi = OUTPUT_RANGE
    n = len(trace["aggregated"]) - 1
    xs = [lo + (hi - lo) * k / n for k in range(n + 1)]
    ys = trace["aggregated"]
    ax.plot(xs, ys, color=SKY, linewidth=2)
    ax.fill_between(xs, ys, color=SKY, alpha=0.18)
    ax.axvline(utility, color=GREEN, linewidth=2, linestyle="--",
               label=f"centroide = {utility:.1f}")
    ax.set_title("Salida difusa agregada y defuzzificacion (centroide)", fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
    ax.set_xlabel("utilidad")
    ax.legend(fontsize=8, facecolor="none", labelcolor=TEXT, edgecolor=GRID)
    _style(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3. Curva de robustez vs sigma (cobertura media +- desviacion)
# ----------------------------------------------------------------------
def fig_robustness(result: dict, algorithm: str):
    from src.simulation import robustness_curve
    fig, ax = _fig(7.5, 3.4)
    for policy in ("lineal", "difuso"):
        cur = robustness_curve(result, policy, algorithm)
        sig = cur["sigmas"]
        mean = cur["cov_mean"]
        std = [v ** 0.5 for v in cur["cov_var"]]
        color = POLICY_COLOR[policy]
        ax.plot(sig, mean, color=color, linewidth=2.2, marker="o", label=policy)
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(sig, lo, hi, color=color, alpha=0.15)
    # A* control: linea horizontal (independiente del ruido)
    astar_cov = result.get("astar_cov")
    if astar_cov is not None:
        ax.axhline(astar_cov, color=GREEN, linestyle="--", linewidth=1.8,
                   label=f"A* (control = {astar_cov:.0f}%)")
    ax.set_title(f"Robustez vs ruido - {algorithm}", fontsize=11)
    ax.set_xlabel("sigma (intensidad de ruido del LLM)")
    ax.set_ylabel("cobertura media (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, facecolor="none", labelcolor=TEXT, edgecolor=GRID)
    _style(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 4. Histogramas de cobertura a un sigma dado (lineal vs difuso)
# ----------------------------------------------------------------------
def fig_histograms(result: dict, sigma: float, algorithm: str, metric: str = "cov"):
    from src.simulation import get_record
    key = "cov_samples" if metric == "cov" else "hours_samples"
    fig, ax = _fig(7.5, 3.2)
    for policy in ("lineal", "difuso"):
        rec = get_record(result, sigma, policy, algorithm)
        if not rec:
            continue
        samples = rec[key]
        ax.hist(samples, bins=15, color=POLICY_COLOR[policy], alpha=0.55,
                label=f"{policy} (media {rec[metric + '_mean']:.1f})", edgecolor="none")
    astar = result.get("astar_cov" if metric == "cov" else "astar_hours")
    if astar is not None:
        ax.axvline(astar, color=GREEN, linewidth=2, linestyle="--",
                   label=f"A* = {astar:.0f}")
    unit = "cobertura (%)" if metric == "cov" else "horas"
    ax.set_title(f"{algorithm} - distribucion de {unit} (sigma={sigma})", fontsize=10)
    ax.set_xlabel(unit); ax.set_ylabel("frecuencia")
    ax.legend(fontsize=8, facecolor="none", labelcolor=TEXT, edgecolor=GRID)
    _style(ax)
    fig.tight_layout()
    return fig
