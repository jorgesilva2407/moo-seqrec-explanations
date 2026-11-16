"""Build the article's figures (report/src/figuras/*.pdf) and table bodies (report/src/tabelas/*.tex)
from the committed result files (results/*.json) and report/dados/exemplo.json.

    uv run --group explain python report/scripts/figuras.py        (or: just report-figures)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
RESULTS = REPORT.parent / "results"
FIGURES = REPORT / "src" / "figuras"
TABLES = REPORT / "src" / "tabelas"

NAMES = {
    "nsga2": "NSGA-II",
    "nsga2_seeded": "NSGA-II semeado$^\dagger$",
    "scalar_ga@main-sga-p25-d4": "AG escalarizado (25, 15)$^\dagger$",
    "scalar_ga@main-sga-p10-d2": "AG escalarizado (10, 6)$^\dagger$",
    "greedy": "Guloso",
    "scalar_ga": "AG escalarizado (20, 10)",
    "ablation": "Ranking por ablação",
    "gece_restarts": "GECE com reinícios",
    "gece": "GECE",
    "recency": "Recência",
    "random": "Aleatório",
    "similarity": "Similaridade",
    "enumeration": r"Enumeração ($|X|\le 2$)",
}
# One colour and one marker per method, the same in every figure (colour is never the only cue).
STYLE = {
    "nsga2": ("#2a78d6", "o", "-"),
    "greedy": ("#eb6834", "s", "--"),
    "scalar_ga": ("#1baf7a", "^", "-."),
    "gece_restarts": ("#eda100", "D", ":"),
    "ablation": ("#e87ba4", "v", "--"),
    "nsga2_seeded": ("#008300", "P", "-"),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
COLUMN = 3.33  # width of one acmart sigconf column, in inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def num(value: float, digits: int = 3) -> str:
    """Decimal comma, as in the text."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", r"\,")


def write_table(name: str, rows: list[list[str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    (TABLES / name).write_text("%% Gerado por report/scripts/figuras.py; não editar.\n" + body + "\n", encoding="utf-8")


def plot(ax, x, y, method: str, **kw) -> None:
    colour, marker, line = STYLE[method]
    ax.plot(x, y, color=colour, marker=marker, linestyle=line, label=NAMES[method], **kw)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


# ---------------------------------------------------------------- figures


def convergence() -> None:
    """Hypervolume along the search: masks of at most 10 positions (left) and of every size (right)."""
    data = load("main-final-method.json")
    short = {"nsga2_seeded": "NSGA-II semeado", "scalar_ga": "AG escalarizado"}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.75), sharex=True)
    for axis, key, title in ((axes[0], "convergence_le10", r"$|X|\leq 10$"), (axes[1], "convergence_all", "todos os tamanhos")):
        for method in ("nsga2", "nsga2_seeded", "greedy", "ablation", "scalar_ga"):
            colour, marker, line = STYLE[method]
            axis.plot(data[key]["checkpoints"], data[key]["curves"][method], color=colour, marker=marker, linestyle=line,
                      markersize=2.6, linewidth=1.1, label=short.get(method, NAMES[method]))
        axis.set_xscale("log")
        axis.set_xlabel("máscaras avaliadas")
        axis.set_title(title, fontsize=7)
        axis.set_ylim(0, 1.15)
    axes[0].set_ylabel("hipervolume")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, columnspacing=1.0, handlelength=2.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "convergencia.pdf")


def example_front() -> None:
    """The NSGA-II front of the worked example in the (N, S) plane, coloured by size."""
    with open(REPORT / "dados" / "exemplo.json", encoding="utf-8") as f:
        example = json.load(f)
    front = np.asarray(example["fronts"]["nsga2"])
    front = front[front[:, 2] <= 10]
    chosen = example["explanation"]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#b7d3f6", "#2a78d6", "#0d366b"])
    points = ax.scatter(front[:, 0], front[:, 1], c=front[:, 2], cmap=ramp, vmin=1, vmax=10, s=14,
                        edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter([chosen["N"]], [chosen["S"]], s=60, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=4)
    ax.annotate("explicação do exemplo", (chosen["N"], chosen["S"]), xytext=(0.30, 0.86), textcoords="axes fraction",
                ha="center", fontsize=6.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkB": 5})
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel(r"$\hat N$ (menor = mais necessário)")
    ax.set_ylabel(r"$\hat S$ (menor = mais suficiente)")
    bar = fig.colorbar(points, ax=ax, pad=0.02, ticks=[1, 4, 7, 10])
    bar.set_label("tamanho $|X|$")
    bar.outline.set_visible(False)
    save(fig, "exemplo-frente.pdf")


def correlation() -> None:
    data = load("main-final-method.json")["objective_correlation"]
    labels = ["0–10", "10–25", "25–50", "50–75", "75–100"]
    values = list(data["by_size"].values())
    fig, ax = plt.subplots(figsize=(COLUMN, 1.6))
    bars = ax.bar(labels, values, width=0.6, color="#2a78d6")
    ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in values], fontsize=6.5, color=INK, padding=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tamanho da máscara (% de $n$)")
    ax.set_ylabel(r"Spearman entre $\hat N$ e $\hat S$")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    save(fig, "correlacao.pdf")


# ---------------------------------------------------------------- tables


def methods_table() -> None:
    data = load("main-final-method.json")
    summary, quality = data["summary"], data["quality"]
    order = sorted(summary, key=lambda m: -summary[m]["hv_le10_mean"])
    best_all = max(summary[m]["hv_mean"] for m in order)
    best_small = max(summary[m]["hv_le10_mean"] for m in order)
    bold = lambda value, best: rf"\textbf{{{num(value)}}}" if value == best else num(value)  # noqa: E731
    write_table("metodos.tex", [
        [NAMES[m], bold(summary[m]["hv_le10_mean"], best_small), bold(summary[m]["hv_mean"], best_all),
         num(summary[m]["front_size"], 1), thousands(summary[m]["passes"]),
         num(quality[m]["best_pn"], 2), num(quality[m]["best_ps"], 2)]
        for m in order
    ])


def grid_table(name: str, source: str, rows: list, columns: list, key, field: str) -> None:
    summary = load(source)["summary"]
    best = max(summary[key(r, c)][field] for r, _ in rows for c, _ in columns)
    cell = lambda v: rf"\textbf{{{num(v)}}}" if v == best else num(v)  # noqa: E731
    write_table(name, [[label] + [cell(summary[key(r, c)][field]) for c, _ in columns] for r, label in rows])


def operators_table() -> None:
    summary = load("nsga2-operators-stage.json")["summary"]
    labels = [
        ("tune-p25-m10", r"Configuração ajustada"),
        ("op-mut0.5", r"Mutação $0{,}5/n$"),
        ("op-mut2", r"Mutação $2/n$"),
        ("op-mut4", r"Mutação $4/n$"),
        ("op-no-crossover", "Sem cruzamento"),
        ("op-init-uniform", "Inicialização uniforme"),
    ]
    write_table("operadores.tex", [
        [label, num(summary[key]["hv_le10_mean"]), num(summary[key]["hv_mean"]), num(summary[key]["front_size"], 1)]
        for key, label in labels
    ])


def ablation_table() -> None:
    summary = load("background-ablation.json")["summary"]
    fills = (("main", "popularidade"), ("abl-users", "outros usuários"), ("abl-padding", r"\emph{padding}"))
    rows = []
    for method in ("nsga2", "greedy", "ablation", "scalar_ga"):
        for i, (stage, label) in enumerate(fills):
            row = summary[f"{stage}:{method}"]
            rows.append([NAMES[method] if i == 0 else "", label, num(row["hv_le10_mean"]), num(row["hv_mean"])])
    write_table("ablacao.tex", rows)


def example_table() -> None:
    with open(REPORT / "dados" / "exemplo.json", encoding="utf-8") as f:
        example = json.load(f)
    escape = lambda text: text.replace("&", r"\&")  # noqa: E731
    write_table("exemplo.tex", [[str(p), escape(example["history"][p - 1])] for p in example["explanation"]["positions"]])


def exact_front_table() -> None:
    data = load("exact-front.json")
    pct = lambda v: num(100 * v, 1)  # noqa: E731
    rows = []
    for c, row in data["summary"]["nsga2"].items():
        other = data["summary"]["random"][c]
        share = num(100 * int(c) / data["config"]["masks"], 0)
        rows.append([thousands(int(c)), share, pct(row["hv_ratio_mean"]), pct(row["front_found_mean"]),
                     pct(other["hv_ratio_mean"]), pct(other["front_found_mean"])])
    write_table("exata.tex", rows)


def main() -> None:
    exact_front_table()
    example_table()
    convergence()
    example_front()
    correlation()
    methods_table()
    pops, ms = [(p, str(p)) for p in (25, 50, 100, 200)], [(m, str(m)) for m in (5, 10, 20)]
    grid_table("ajuste.tex", "tuning-stage.json", pops, ms, lambda p, m: f"tune-p{p}-m{m}", "hv_mean")
    grid_table("ajuste-pequenas.tex", "tuning-stage.json", pops, ms, lambda p, m: f"tune-p{p}-m{m}", "hv_le10_mean")
    weights = [(d, w) for d, w in ((2, "6"), (3, "10"), (4, "15"))]
    spops = [(p, str(p)) for p in (10, 25, 50)]
    grid_table("escalar.tex", "scalar-ga-tuning-stage.json", spops, weights, lambda p, d: f"sga-p{p}-d{d}", "hv_mean")
    grid_table("escalar-pequenas.tex", "scalar-ga-tuning-stage.json", spops, weights, lambda p, d: f"sga-p{p}-d{d}", "hv_le10_mean")
    operators_table()
    ablation_table()
    print(f"figures in {FIGURES}, tables in {TABLES}")


if __name__ == "__main__":
    main()
