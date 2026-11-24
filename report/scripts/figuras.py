"""Build the article's figures (report/src/figuras/*.pdf), table bodies (report/src/tabelas/*.tex) and
the result numbers quoted in the prose (report/src/tabelas/numeros.tex, used as \\val{key}) from the
committed result files (results/*.json) and report/dados/exemplo.json. Fixed facts of the setup (dataset
sizes, budgets, grids) are written in the text; every measured number comes from here.

    uv run --group explain python report/scripts/figuras.py        (or: just report-figures)
"""

from __future__ import annotations

import json
import math
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

POST_HOC = r"$^\dagger$"
NAMES = {
    "nsga2": "NSGA-II",
    "nsga2_seeded": "NSGA-II semeado" + POST_HOC,
    "nsga2_nc": r"NSGA-II só $(N,C)$" + POST_HOC,
    "smsemoa": "SMS-EMOA" + POST_HOC,
    "nsga3": "NSGA-III" + POST_HOC,
    "moead": "MOEA/D" + POST_HOC,
    "shapley": "Shapley (amostragem)" + POST_HOC,
    "scalar_ga": "AG escalarizado (20, 10)",
    "scalar_ga@main-sga-p25-d4": "AG escalarizado (25, 15)" + POST_HOC,
    "scalar_ga@main-sga-p10-d2": "AG escalarizado (10, 6)" + POST_HOC,
    "greedy": "Guloso",
    "ablation": "Ranking por ablação",
    "gece_restarts": "GECE com reinícios",
    "gece": "GECE",
    "recency": "Recência",
    "random": "Aleatório",
    "similarity": "Similaridade",
    "enumeration": r"Enumeração ($|X|\le 2$)",
}
SHORT = {"nsga2_seeded": "NSGA-II semeado", "scalar_ga": "AG escalarizado", "smsemoa": "SMS-EMOA", "shapley": "Shapley",
         "nsga3": "NSGA-III", "moead": "MOEA/D", "nsga2_nc": "NSGA-II só (N,C)"}
# One colour, marker and dash per method, the same in every figure (colour is never the only cue).
STYLE = {
    "nsga2": ("#2a78d6", "o", "-"),
    "greedy": ("#eb6834", "s", "--"),
    "scalar_ga": ("#1baf7a", "^", "-."),
    "gece_restarts": ("#eda100", "D", ":"),
    "ablation": ("#e87ba4", "v", "--"),
    "nsga2_seeded": ("#008300", "P", "-"),
    "smsemoa": ("#6250d6", "X", "-"),
    "shapley": ("#e34948", "*", ":"),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
COLUMN = 3.33  # width of one acmart sigconf column, in inches
REPORTS = {"ml": "final-ml20m-method.json", "rd": "final-redraw-method.json", "zv": "final-zvuk-method.json",
           "za": "final-zvuk-alt-method.json"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

VALUES: dict[str, str] = {}


def load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def num(value: float, digits: int = 3) -> str:
    """Decimal comma, as in the text."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def signed(value: float, digits: int = 3) -> str:
    return ("+" if value >= 0 else "$-$") + num(abs(value), digits)


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", r"\,")


def pvalue(p: float) -> str:
    """0,023 / 3,3x10^-5 / <10^-15, wrapped in \\ensuremath so it works inside $p=...$ and in tables."""
    if p >= 1e-3:
        return num(p, 3)
    if p < 1e-15:
        return r"\ensuremath{<10^{-15}}"
    exponent = math.floor(math.log10(p))
    return rf"\ensuremath{{{num(p / 10**exponent, 1)}\times10^{{{exponent}}}}}"  # works in and out of math


def key(name: str) -> str:
    """Method ids as macro keys (no underscores or @)."""
    return name.replace("scalar_ga@main-sga-", "sga-").replace("_", "-")


def val(name: str, text: str) -> None:
    VALUES[name] = text


def write_table(name: str, rows: list[list[str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    (TABLES / name).write_text("%% Gerado por report/scripts/figuras.py; não editar.\n" + body + "\n", encoding="utf-8")


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


# ---------------------------------------------------------------- numbers quoted in the prose


def report_values() -> None:
    """Per report (ml, rd, zv, za): every method's summary, every paired test against the best label
    and every extra pair; keys like ml.nsga2.hvle, ml.pair.smsemoa.nsga2.le.wins."""
    for tag, file in REPORTS.items():
        data = load(file)
        for method, row in data["summary"].items():
            k = f"{tag}.{key(method)}"
            val(f"{k}.hv", num(row["hv_mean"]))
            val(f"{k}.hvle", num(row["hv_le10_mean"]))
            val(f"{k}.front", num(row["front_size"], 1))
            val(f"{k}.passes", thousands(row["passes"]))
            for field, short in (("igd_plus_mean", "igd"), ("igd_plus_le10_mean", "igdle"), ("hv_le10_nc_mean", "hvlenc")):
                if field in row:
                    val(f"{k}.{short}", num(row[field], 3))
            q = data["quality"].get(method)
            if q:
                for stat in ("best_pn", "best_ps", "recency", "valid"):
                    val(f"{k}.{stat.replace('best_', '')}", num(q[stat], 2))
        val(f"{tag}.instances", str(next(iter(data["summary"].values()))["instances"]))
        for region, short in (("wilcoxon_hv", "all"), ("wilcoxon_small", "le")):
            best = data[region]["best"]
            val(f"{tag}.best.{short}", key(best))
            for other, t in data[region]["against"].items():
                test_values(f"{tag}.vs.{short}.{key(best)}.{key(other)}", t)
        for pair, fields in data.get("pairs", {}).items():
            a, b = pair.split(",")
            for field, t in fields.items():
                test_values(f"{tag}.pair.{key(a)}.{key(b)}.{'all' if field == 'hv' else 'le'}", t)
        popularity = data.get("small_hv_by_target_popularity", {}).get("nsga2")
        if popularity:
            for band, v in zip(("low", "mid", "high"), popularity):
                val(f"{tag}.nsga2.pop.{band}", num(v, 2))


STAGE_REPORTS = {"tn": "tuning-stage.json", "sg": "scalar-ga-tuning-stage.json", "op": "nsga2-operators-stage.json",
                 "mt": "new-moea-tuning-stage.json"}


def stage_values() -> None:
    """Tuning-type reports (--by stage): every stage's summary and tests against the best stage;
    keys like tn.tune-p25-m10.hv and tn.vs.all.tune-p25-m10.tune-p100-m10.wins."""
    for tag, file in STAGE_REPORTS.items():
        data = load(file)
        for stage, row in data["summary"].items():
            val(f"{tag}.{stage}.hv", num(row["hv_mean"]))
            val(f"{tag}.{stage}.hvle", num(row["hv_le10_mean"]))
        for region, short in (("wilcoxon_hv", "all"), ("wilcoxon_small", "le")):
            best = data[region]["best"]
            for other, t in data[region]["against"].items():
                test_values(f"{tag}.vs.{short}.{best}.{other}", t)
        values = sorted(row["hv_le10_mean"] for row in data["summary"].values())
        val(f"{tag}.hvle.min", num(values[0], 2))
        val(f"{tag}.hvle.second", num(values[1], 2))
        val(f"{tag}.hvle.max", num(values[-1], 2))


def test_values(prefix: str, t: dict) -> None:
    val(f"{prefix}.wins", str(t["wins"]))
    val(f"{prefix}.losses", str(t["losses"]))
    val(f"{prefix}.n", str(t["instances"]))
    val(f"{prefix}.p", pvalue(t.get("p_holm", t["p"])))
    if "mean_diff" in t:
        val(f"{prefix}.diff", signed(t["mean_diff"]))
        val(f"{prefix}.ci", f"[{signed(t['ci95'][0])}; {signed(t['ci95'][1])}]")
        val(f"{prefix}.r", signed(t["rank_biserial"], 2))


def other_values() -> None:
    corr = load("final-ml20m-method.json")["objective_correlation"]
    val("corr.overall", num(corr["overall"], 2))
    val("corr.size1", num(corr["size_1"], 2))
    val("corr.size2", num(corr["size_2"], 2))
    for (label, v), name in zip(corr["by_size"].items(), ("b1", "b2", "b3", "b4", "b5")):
        val(f"corr.{name}", num(v, 2))
    m100 = load("conflict-m100.json")
    for size in (1, 2):
        for m in (10, 100):
            val(f"m100.size{size}.m{m}", num(m100[f"size_{size}"][f"M={m}"]["mean"], 2))
    checks = load("offline-checks.json")
    share = checks["share_of_evaluated_masks_le10"]
    val("seedcost.nsga2", num(100 * share["nsga2"], 0))
    val("seedcost.seeded", num(100 * share["nsga2_seeded"], 0))
    ps = checks["smallest_size_with_ps_ge_half"]
    val("psmin.median", num(ps["median"], 0))
    val("psmin.q1", num(ps["quartiles"][0], 0))
    val("psmin.q3", num(ps["quartiles"][1], 0))
    val("psmin.le10", num(100 * ps["share_le10"], 0))
    val("psmin.rel", num(100 * ps["median_relative"], 0))
    val("psmin.never", str(ps["never"]))
    bands = checks["spread_by_size_bin"]
    for b, name in zip(bands, ("b1", "b2", "b3", "b4", "b5")):
        val(f"snr.{name}.n", num(bands[b]["necessity"]["spread_over_noise"], 1))
        val(f"snr.{name}.s", num(bands[b]["sufficiency"]["spread_over_noise"], 1))
    rescored = load("ps-rescored.json")
    val("psre.search", num(rescored["search_ps_mean"], 2))
    val("psre.eval", num(rescored["eval_ps_mean"], 2))
    val("psre.kept", num(100 * rescored["share_still_sufficient"], 0))
    val("psre.keptle", num(100 * rescored["share_still_sufficient_le10"], 0))
    for name, file in (("ml", "bench-ml20m.json"), ("zv", "bench-zvuk.json")):
        bench = load(file)
        agreement = bench["agreement_with_reference"]["10"]
        val(f"bench.{name}.n", num(agreement["necessity_spearman"], 2))
        val(f"bench.{name}.s", num(agreement["sufficiency_spearman"], 2))
        val(f"bench.{name}.speed", thousands(1000 * round(max(v["passes_per_second"] or 0 for v in bench["throughput"].values()) / 1000)))
    zvuk_conflict = RESULTS / "conflict-m100-zvuk.json"
    if zvuk_conflict.exists():
        zc = load(zvuk_conflict.name)
        for size in (1, 2):
            for m in (10, 100):
                val(f"m100zv.size{size}.m{m}", num(zc[f"size_{size}"][f"M={m}"]["mean"], 2))
    for method, row in load("search-size-profile.json").items():
        val(f"size.{key(method)}.le10", num(100 * row["share_evaluated_le10"], 0))
        val(f"size.{key(method)}.median", num(100 * row["median_relative_size_evaluated"], 0))
        val(f"size.{key(method)}.first", num(row["median_size_first_21"], 1))
    exact = load("exact-front-n16.json")
    val("exact.instances", str(exact["instances"]))
    val("exact.front", num(exact["exact_front_size_mean"], 0))
    val("exact.masks", thousands(exact["config"]["masks"]))
    for method, per in exact["summary"].items():
        for c, v in per.items():
            val(f"exact.{key(method)}.{c}.hv", num(100 * v["hv_ratio_mean"], 1))
            val(f"exact.{key(method)}.{c}.found", num(100 * v["front_found_mean"], 0))
    gece = load("gece-returned.json")["gece"]
    val("gece.queries", num(gece["queries_mean"], 0))
    val("gece.single", num(100 * gece["returned_single_position_share"], 0))
    val("gece.pn", num(gece["returned_pn_mean"], 2))
    val("gece.necessary", num(100 * gece["returned_pn_above_half_share"], 0))
    val("gece.valid", str(gece["valid"]))
    val("gece.jobs", str(gece["jobs"]))
    models = load("model-checks.json")
    for ds, short in (("ml-20m", "ml"), ("zvuk", "zv")):
        m = models[ds]
        for method in ("sasrec", "sasrec_shuffled", "popularity", "last_item_transitions"):
            val(f"model.{short}.{key(method)}", num(m[method]["ndcg@10"], 3))
        val(f"model.{short}.drop", num(100 * m["shuffle_drop_ndcg@10"], 0))
        val(f"model.{short}.margin", num(100 * (m["sasrec"]["ndcg@10"] / m["last_item_transitions"]["ndcg@10"] - 1), 0))
    val("model.users", thousands(models["zvuk"]["users"]))
    training = load("sasrec-training.json")
    ml = training["ml-20m"]
    val("train.ml.test.hr", num(ml["test"]["hr@10"], 3))
    val("train.ml.test.ndcg", num(ml["test"]["ndcg@10"], 3))
    val("train.ml.best", str(ml["best_epoch"]))
    val("train.ml.epochs", str(ml["epochs_run"]))
    selected = training["zvuk-final"][training["zvuk-selected"]]
    val("train.zv.test.hr", num(selected["test"]["hr@10"], 3))
    val("train.zv.test.ndcg", num(selected["test"]["ndcg@10"], 3))
    val("train.zv.val.ndcg", num(selected["val"]["ndcg@10"], 3))
    val("train.zv.best", str(selected["best_epoch"]))
    val("train.zv.epochs", str(selected["epochs_run"]))
    search = training["zvuk-search"]
    values = sorted(r["val"]["ndcg@10"] for r in search.values())
    val("train.zv.search.min", num(values[0], 3))
    val("train.zv.search.max", num(values[-1], 3))
    finals = sorted(r["val"]["ndcg@10"] for r in training["zvuk-final"].values())
    val("train.zv.final.spread", num(finals[-1] - finals[0], 3))
    tuning = load("new-moea-tuning-stage.json")["summary"]
    for stage, row in tuning.items():
        val(f"tune.{stage}.hv", num(row["hv_mean"]))
        val(f"tune.{stage}.hvle", num(row["hv_le10_mean"]))
    small = load("new-moea-tuning-stage.json")["wilcoxon_small"]
    if "tune-p25-m10" in small["against"]:
        test_values(f"tune.vs.le.{small['best']}.nsga2", small["against"]["tune-p25-m10"])
        val("tune.best.le", small["best"])
    seeded = load("nsga2-seeded-tuning-stage.json")
    for region, short in (("wilcoxon_hv", "all"), ("wilcoxon_small", "le")):
        t = next(iter(seeded[region]["against"].values()))
        test_values(f"seedtune.{short}", t)


def write_values() -> None:
    lines = ["%% Gerado por report/scripts/figuras.py; não editar. Uso no texto: \\val{chave}."]
    lines += [rf"\valdef{{{k}}}{{{v}}}" for k, v in sorted(VALUES.items())]
    (TABLES / "numeros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- figures


def convergence() -> None:
    """Hypervolume along the search: masks of at most 10 positions (left) and of every size (right)."""
    data = load("final-ml20m-method.json")
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.85), sharex=True)
    for axis, k, title in ((axes[0], "convergence_le10", r"$|X|\leq 10$"), (axes[1], "convergence_all", "todos os tamanhos")):
        for method in ("nsga2", "nsga2_seeded", "smsemoa", "greedy", "shapley", "scalar_ga"):
            colour, marker, line = STYLE[method]
            axis.plot(data[k]["checkpoints"], data[k]["curves"][method], color=colour, marker=marker, linestyle=line,
                      markersize=2.6, linewidth=1.1, label=SHORT.get(method, NAMES[method]))
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


FAMILY = {  # colour of each method in the key figure, by how it searches
    "smsemoa": "#2a78d6", "nsga3": "#2a78d6", "nsga2": "#2a78d6", "nsga2_seeded": "#2a78d6", "nsga2_nc": "#2a78d6",
    "moead": "#1baf7a", "scalar_ga": "#1baf7a",
    "greedy": "#eb6834", "shapley": "#eb6834", "ablation": "#eb6834",
    "gece_restarts": "#eda100",
}
SCENARIOS = (("ml", "ML-20M principal", "o"), ("rd", "ML-20M novo", "s"), ("zv", "Zvuk", "^"))


def key_figure() -> None:
    """The main story at a glance: (a) N and S agree only moderately on small masks; (b) paired HV<=10 gap of
    every method to the best one (SMS-EMOA in the three scenarios) with its bootstrap CI; (c) NSGA-II with and
    without S in the (N, C) projection and in the full HV<=10."""
    ml = load("final-ml20m-method.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1.0, 1.75, 1.0]})

    ax = axes[0]
    labels = ["0–10", "10–25", "25–50", "50–75", "75–100"]
    values = list(ml["objective_correlation"]["by_size"].values())
    bars = ax.bar(labels, values, width=0.6, color=["#2a78d6"] + ["#b7d3f6"] * 4)
    ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in values], fontsize=6, color=INK, padding=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tamanho da máscara (% de $n$)")
    ax.set_ylabel(r"Spearman entre $\hat N$ e $\hat S$")
    ax.set_title("(a) conflito nas máscaras pequenas", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax = axes[1]
    reports = {tag: load(REPORTS[tag]) for tag, _, _ in SCENARIOS}
    reference = reports["ml"]["wilcoxon_small"]["best"]
    assert all(r["wilcoxon_small"]["best"] == reference for r in reports.values())
    methods = ("nsga3", "nsga2", "nsga2_seeded", "nsga2_nc", "greedy", "shapley", "ablation", "scalar_ga",
               "gece_restarts", "moead")
    for row, method in enumerate(methods):
        for offset, (tag, label, marker) in zip((-0.22, 0.0, 0.22), SCENARIOS):
            t = reports[tag]["wilcoxon_small"]["against"].get(method)
            if t is None:
                continue
            y = row + offset
            ax.plot(t["ci95"], [y, y], color=FAMILY[method], linewidth=1.0)
            ax.plot(t["mean_diff"], y, marker=marker, color=FAMILY[method], markersize=3.2, linestyle="none",
                    markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_yticks(range(len(methods)), [SHORT.get(m, NAMES[m]).replace(POST_HOC, "") for m in methods])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.05, linscale=1.0)
    ax.set_xticks([0, 0.02, 0.05, 0.2, 0.5], ["0", "0,02", "0,05", "0,2", "0,5"])
    ax.set_xlabel(rf"HV$_{{\leq 10}}$({SHORT[reference]}) $-$ HV$_{{\leq 10}}$(método)")
    ax.set_title(f"(b) distância ao melhor método ({SHORT[reference]}), IC 95%", fontsize=7)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker=m, color=MUTED, linestyle="none", markersize=3.2) for _, _, m in SCENARIOS]
    ax.legend(handles, [label for _, label, _ in SCENARIOS], loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, handletextpad=0.2)

    ax = axes[2]
    summary = ml["summary"]
    groups = (("hv_le10_nc_mean", r"proj. $(N,C)$"), ("hv_le10_mean", r"$(N,S,C)$"))
    for i, (field, _) in enumerate(groups):
        for j, (method, colour) in enumerate((("nsga2", "#2a78d6"), ("nsga2_nc", "#b7d3f6"))):
            v = summary[method][field]
            b = ax.bar(i + (j - 0.5) * 0.42, v, width=0.4, color=colour,
                       label=SHORT.get(method, NAMES[method]).replace(POST_HOC, "") if i == 0 else None)
            ax.bar_label(b, labels=[num(v, 3).replace("{,}", ",")], fontsize=5, color=INK, padding=1)
    ax.set_xticks(range(len(groups)), [label for _, label in groups])
    ax.set_ylim(0.8, 1.04)
    ax.set_ylabel(r"HV$_{\leq 10}$")
    ax.set_title("(c) com e sem $S$", fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, handlelength=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "principal.pdf")


# ---------------------------------------------------------------- tables


def methods_table() -> None:
    """Table 3 split in two: the methods of the pre-planned comparison and the follow-up experiments (marked
    with a dagger), both from the same report and so with the same per-instance normalisation; NSGA-II is
    repeated as the reference row of the second. IGD+, PN and PS of every method go to an appendix table."""
    data = load("final-ml20m-method.json")
    summary, quality = data["summary"], data["quality"]
    replicated = {tag: load(REPORTS[tag])["summary"] for tag in ("rd", "zv")}
    order = sorted(summary, key=lambda m: -summary[m]["hv_le10_mean"])
    follow_up = [m for m in order if NAMES[m].endswith(POST_HOC)]
    planned = [m for m in order if m not in follow_up]

    def rows(methods: list[str], extra) -> list[list[str]]:
        best = {f: max(summary[m][f] for m in methods) for f in ("hv_mean", "hv_le10_mean")}
        cell = lambda m, f: rf"\textbf{{{num(summary[m][f])}}}" if summary[m][f] == best[f] else num(summary[m][f])  # noqa: E731
        return [[NAMES[m], cell(m, "hv_le10_mean"), cell(m, "hv_mean"), num(summary[m]["front_size"], 1),
                 thousands(summary[m]["passes"])] + extra(m) for m in methods]

    write_table("metodos-plan.tex", rows(planned, lambda m: []))
    where = lambda m: [", ".join(label for tag, label in (("rd", "ML novo"), ("zv", "Zvuk")) if m in replicated[tag]) or "--"]  # noqa: E731
    write_table("metodos-pos.tex", rows(["nsga2"] + follow_up, where))
    best_igd = min(summary[m]["igd_plus_le10_mean"] for m in order)
    igd = lambda m: rf"\textbf{{{num(summary[m]['igd_plus_le10_mean'])}}}" if summary[m]["igd_plus_le10_mean"] == best_igd else num(summary[m]["igd_plus_le10_mean"])  # noqa: E731
    write_table("metodos-extra.tex", [
        [NAMES[m], igd(m), num(quality[m]["best_pn"], 2), num(quality[m]["best_ps"], 2), num(quality[m]["recency"], 2)]
        for m in order
    ])


PLANNED_PAIRS = [  # (a, b, region): a is better than b in that region (as reported); methods of the planned comparison
    ("nsga2", "scalar_ga", "le"), ("nsga2", "greedy", "le"), ("greedy", "nsga2", "all"),
]
FOLLOW_UP_PAIRS = [  # pairs with at least one follow-up method
    ("smsemoa", "nsga2", "le"), ("nsga3", "nsga2", "le"), ("nsga2", "moead", "le"), ("moead", "nsga2", "all"),
    ("moead", "scalar_ga", "all"), ("scalar_ga", "moead", "le"),
    ("nsga2", "shapley", "le"), ("shapley", "nsga2", "all"),
    ("nsga2", "nsga2_nc", "le"), ("nsga2", "nsga2_nc", "all"),
    ("nsga2_seeded", "greedy", "all"), ("nsga2", "nsga2_seeded", "le"),
]


def pairs_table() -> None:
    pairs = load("final-ml20m-method.json")["pairs"]
    label = lambda m: SHORT.get(m, NAMES[m].replace(POST_HOC, "")) + (POST_HOC if NAMES[m].endswith(POST_HOC) else "")  # noqa: E731
    for name, chosen in (("pares-plan.tex", PLANNED_PAIRS), ("pares-pos.tex", FOLLOW_UP_PAIRS)):
        rows = []
        for a, b, region in chosen:
            t, flip = pairs.get(f"{a},{b}"), False
            if t is None:
                t, flip = pairs[f"{b},{a}"], True
            t = t["hv" if region == "all" else "hv_le10"]
            wins, diff, lo, hi, r = t["wins"], t["mean_diff"], t["ci95"][0], t["ci95"][1], t["rank_biserial"]
            if flip:
                wins, diff, lo, hi, r = t["losses"], -diff, -hi, -lo, -r
            rows.append([f"{label(a)} $>$ {label(b)}", r"\HVp" if region == "le" else "HV", f"{wins}/{t['instances']}",
                         f"{signed(diff)} [{signed(lo)}; {signed(hi)}]", signed(r, 2), pvalue(t["p_holm"])])
        write_table(name, rows)


ROBUST_METHODS = ("nsga2", "nsga2_seeded", "smsemoa", "moead", "greedy", "ablation", "scalar_ga")


def robustness_table() -> None:
    """Core methods in the main set, the new set and Zvuk (main text); Zvuk with the second-best model (appendix)."""
    data = {tag: load(file)["summary"] for tag, file in REPORTS.items()}

    def rows(tags: tuple[str, ...]) -> list[list[str]]:
        out = []
        for method in ROBUST_METHODS:
            if not any(method in data[tag] for tag in tags):
                continue
            row = [SHORT.get(method, NAMES[method].replace(POST_HOC, ""))]
            for field in ("hv_le10_mean", "hv_mean"):
                for tag in tags:
                    summary = data[tag]
                    if method not in summary:
                        row.append("--")
                        continue
                    v = summary[method][field]
                    top = max(summary[m][field] for m in ROBUST_METHODS if m in summary)
                    row.append(rf"\textbf{{{num(v)}}}" if v == top else num(v))
            out.append(row)
        return out

    write_table("robustez.tex", rows(("ml", "rd", "zv")))
    write_table("robustez-alt.tex", rows(("zv", "za")))


def exact_front_table() -> None:
    data = load("exact-front-n16.json")
    rows = []
    for method in ("smsemoa", "nsga2", "nsga3", "moead", "greedy", "random"):
        per = data["summary"][method]
        rows.append([SHORT.get(method, NAMES[method].replace(POST_HOC, ""))] +
                    [f"{num(100 * per[c]['hv_ratio_mean'], 1)} / {num(100 * per[c]['front_found_mean'], 0)}"
                     for c in ("250", "500", "1000", "2500")])
    write_table("exata.tex", rows)


def moea_tuning_table() -> None:
    summary = load("new-moea-tuning-stage.json")["summary"]
    labels = [
        ("tune-p25-m10", "NSGA-II (pop.\\ 25, referência)"),
        ("tune-smsemoa-p25", "SMS-EMOA, pop.\\ 25"), ("tune-smsemoa-p50", "SMS-EMOA, pop.\\ 50"),
        ("tune-nsga3-p5", "NSGA-III, 21 direções"), ("tune-nsga3-p6", "NSGA-III, 28 direções"),
        ("tune-moead-tch-p5", "MOEA/D Tchebycheff, 21"), ("tune-moead-tch-p6", "MOEA/D Tchebycheff, 28"),
        ("tune-moead-pbi-p5", "MOEA/D PBI, 21"), ("tune-moead-pbi-p6", "MOEA/D PBI, 28"),
    ]
    write_table("moea-ajuste.tex", [[label, num(summary[k]["hv_mean"]), num(summary[k]["hv_le10_mean"])]
                                    for k, label in labels])


def grid_table(name: str, source: str, rows: list, columns: list, keyf) -> None:
    """Rows x columns of configurations: HV (all masks) block, then HV<=10 block; best of each block in bold."""
    summary = load(source)["summary"]
    blocks = []
    for field in ("hv_mean", "hv_le10_mean"):
        best = max(summary[keyf(r, c)][field] for r, _ in rows for c, _ in columns)
        blocks.append((field, best))
    cell = lambda v, best: rf"\textbf{{{num(v)}}}" if v == best else num(v)  # noqa: E731
    write_table(name, [
        [label] + [cell(summary[keyf(r, c)][field], best) for field, best in blocks for c, _ in columns]
        for r, label in rows
    ])


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
        [label, num(summary[k]["hv_le10_mean"]), num(summary[k]["hv_mean"]), num(summary[k]["front_size"], 1)]
        for k, label in labels
    ])
    for k, _ in labels:
        val(f"op.{k}.hvle", num(summary[k]["hv_le10_mean"]))
        val(f"op.{k}.hv", num(summary[k]["hv_mean"]))
    val("op.nocross.drop", num(summary["tune-p25-m10"]["hv_le10_mean"] - summary["op-no-crossover"]["hv_le10_mean"], 2))


def ablation_table() -> None:
    data = load("background-ablation.json")
    summary = data["summary"]
    fills = (("main", "popularidade"), ("abl-users", "outros usuários"), ("abl-padding", r"\emph{padding}"))
    rows = []
    for method in ("nsga2", "greedy", "ablation", "scalar_ga"):
        for i, (stage, label) in enumerate(fills):
            row = summary[f"{stage}:{method}"]
            rows.append([SHORT.get(method, NAMES[method]) if i == 0 else "", label, num(row["hv_le10_mean"]), num(row["hv_mean"])])
    write_table("ablacao.tex", rows)
    for name, v in data["overlap"].items():
        val({"main seeds 0-1 vs 2-4": "overlap.seeds", "main vs abl-users": "overlap.users",
             "main vs abl-padding": "overlap.padding", "abl-users vs abl-padding": "overlap.userspadding"}[name], num(v, 2))
    t = data["tests"]
    val("abl.users.nsga2.all.mainbetter", str(t["hv|nsga2|abl-users"]["main_better"]))
    val("abl.users.nsga2.le.mainbetter", str(t["hv_le10|nsga2|abl-users"]["main_better"]))
    val("abl.padding.min.mainbetter", str(min(t[f"{f}|{m}|abl-padding"]["main_better"] for f in ("hv", "hv_le10")
                                              for m in ("nsga2", "greedy", "ablation", "scalar_ga"))))


def example_table() -> None:
    with open(REPORT / "dados" / "exemplo.json", encoding="utf-8") as f:
        example = json.load(f)
    escape = lambda text: text.replace("&", r"\&")  # noqa: E731
    write_table("exemplo.tex", [[str(p), escape(example["history"][p - 1])] for p in example["explanation"]["positions"]])
    e, single = example["explanation"], example["best_single"]
    val("ex.pn", num(e["PN"], 2))
    val("ex.ps", num(e["PS"], 2))
    val("ex.pnpct", num(100 * e["PN"], 0))
    val("ex.pspct", num(100 * e["PS"], 0))
    val("ex.single.pnpct", num(100 * single["PN"], 0))
    val("ex.single.pspct", num(100 * single["PS"], 0))
    val("ex.n", str(example["n"]))


def training_tables() -> None:
    training = load("sasrec-training.json")
    search, final = training["zvuk-search"], training["zvuk-final"]
    order = sorted(search, key=lambda n: -search[n]["val"]["ndcg@10"])
    rows = []
    for name in order:
        r = search[name]
        f = final.get(name)
        cells = [str(r["hidden_size"]), str(r["num_layers"]), num(r["dropout"], 1), num(r["val"]["ndcg@10"])]
        cells += ([f"{f['epochs_run']} ({f['best_epoch']})", num(f["val"]["ndcg@10"]), num(f["test"]["ndcg@10"])]
                  if f else ["--", "--", "--"])
        if name == training["zvuk-selected"]:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        rows.append(cells)
    write_table("busca-zvuk.tex", rows)
    models = load("model-checks.json")
    labels = (("sasrec", "SASRec"), ("last_item_transitions", "Transições do último item"),
              ("popularity", "Popularidade"), ("sasrec_shuffled", "SASRec, históricos embaralhados"))
    write_table("modelos.tex", [[label] + [f"{num(models[ds][m]['ndcg@10'])} / {num(models[ds][m]['hr@10'])}"
                                           for ds in ("ml-20m", "zvuk")] for m, label in labels])


def main() -> None:
    report_values()
    stage_values()
    other_values()
    exact_front_table()
    example_table()
    convergence()
    example_front()
    key_figure()
    methods_table()
    pairs_table()
    robustness_table()
    moea_tuning_table()
    pops, ms = [(p, str(p)) for p in (25, 50, 100, 200)], [(m, str(m)) for m in (5, 10, 20)]
    grid_table("ajuste.tex", "tuning-stage.json", pops, ms, lambda p, m: f"tune-p{p}-m{m}")
    weights = [(d, w) for d, w in ((2, "6"), (3, "10"), (4, "15"))]
    spops = [(p, str(p)) for p in (10, 25, 50)]
    grid_table("escalar.tex", "scalar-ga-tuning-stage.json", spops, weights, lambda p, d: f"sga-p{p}-d{d}")
    operators_table()
    ablation_table()
    training_tables()
    write_values()
    print(f"figures in {FIGURES}, tables in {TABLES}, {len(VALUES)} values")


if __name__ == "__main__":
    main()
