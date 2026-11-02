"""Paper figures. Reads the judged files through analysis.py so every number
matches the tables. Writes figures/fig_collapse.pdf, fig_form.pdf, fig_turn.pdf."""
import importlib.util, os, sys
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("an", "analysis.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

UNP = {"appeal_to_authority", "circular_reasoning", "hasty_generalization"}
LEVELS = an.LEVELS
MODELS = [("Qwen3-8B", "q8b"), ("Qwen3-32B", "q32b"), ("Gemma4-12B", "g12b"),
          ("Gemma4-31B", "g31b"), ("Llama3.1-8B", "l31_8b")]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK = "#52514e"
STYLE = {  # family = hue; dashed + hollow marker = 8-12B, solid = 31-32B
    "Qwen3-8B":    dict(color=BLUE,   ls="--", marker="o", mfc="white"),
    "Qwen3-32B":   dict(color=BLUE,   ls="-",  marker="o"),
    "Gemma4-12B":  dict(color=ORANGE, ls="--", marker="s", mfc="white"),
    "Gemma4-31B":  dict(color=ORANGE, ls="-",  marker="s"),
    "Llama3.1-8B": dict(color=AQUA,   ls="--", marker="^", mfc="white"),
}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "hatch.linewidth": 0.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5})
W = 4.8      # llncs text width, inches
HALF = 2.33  # two figures side by side

os.makedirs("figures", exist_ok=True)
data = {n: an.valid(an.read_jsonl(f"data/pressure_{t}_judged.jsonl")) for n, t in MODELS}

# Fig 1: collapse rate by level
fig, ax = plt.subplots(figsize=(W, 2.15))
x = list(range(len(LEVELS)))
for name, P in data.items():
    ys, lo, hi = [], [], []
    for lv in LEVELS:
        k, n, _ = an.cell(P, lv, UNP)
        ys.append(k / n); a, b = an.wilson(k, n); lo.append(a); hi.append(b)
    st = STYLE[name]
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.08, linewidth=0)
    ax.plot(x, ys, lw=1.6, ms=4.5, mew=1.2, label=name, **st)
ax.set_xticks(x)
ax.set_xticklabels(["L0\ntrue", "L1\nadjacent", "L2\ncategory", "L3\nviolation",
                    "L4\nimpossible", "L4P\ncontradiction"])
ax.set_ylim(0, 1.02); ax.set_ylabel("Collapse rate"); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.axvspan(3.5, 4.5, color="#000000", alpha=0.03, lw=0)
ax.annotate("content", (4, 0.02), ha="center", va="bottom", fontsize=7, color=INK)
ax.annotate("form", (5, 0.02), ha="center", va="bottom", fontsize=7, color=INK)
ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=2.6)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_collapse.pdf"); plt.close(fig)

# Fig 2: form of first collapse
FAIL = ["SEMANTIC_REFRAME", "FICTIONAL", "EVASIVE", "FACTUAL_YIELD"]
FCOL = {"SEMANTIC_REFRAME": BLUE, "FICTIONAL": ORANGE, "EVASIVE": AQUA, "FACTUAL_YIELD": YELLOW}
FHAT = {"SEMANTIC_REFRAME": "", "FICTIONAL": "..", "EVASIVE": "/", "FACTUAL_YIELD": "x"}
FLAB = {"SEMANTIC_REFRAME": "Reframe", "FICTIONAL": "Fictional", "EVASIVE": "Evasive", "FACTUAL_YIELD": "Yield"}
SHORT = ["Qwen\n8B", "Qwen\n32B", "Gemma\n12B", "Gemma\n31B", "Llama\n8B"]
fig, ax = plt.subplots(figsize=(HALF, 2.2))
names = [n for n, _ in MODELS]; bottoms = [0] * 5
form = {}
for lab in FAIL:
    vals = []
    for name in names:
        labs = [an.first_collapse(r)[1] for r in data[name]
                if r["fallacy"] in UNP and an.first_collapse(r)[0] is not None]
        c = Counter(labs); tot = sum(c.values()); vals.append(c[lab] / tot if tot else 0)
    form[lab] = vals
    ax.bar(SHORT, vals, bottom=bottoms, color=FCOL[lab], hatch=FHAT[lab], edgecolor="white",
           linewidth=0.8, width=0.66, label=FLAB[lab])
    for i, v in enumerate(vals):
        if v >= 0.10:
            ax.text(i, bottoms[i] + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if lab != "FACTUAL_YIELD" else "#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.12", fc=FCOL[lab], ec="none"))
    bottoms = [b + v for b, v in zip(bottoms, vals)]
ax.set_ylim(0, 1); ax.set_ylabel("Share of collapses"); ax.set_yticks([0, .25, .5, .75, 1])
ax.tick_params(axis="x", labelsize=6)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2),
          handlelength=1.3, columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_form.pdf"); plt.close(fig)

# Fig 3: turn of first collapse, by level, pooled over models
LV3 = ["L3", "L4", "L4P"]; SEQ = {"L3": "#a9c8ee", "L4": "#5a96de", "L4P": "#1c4f9c"}
dist = {}
for lv in LV3:
    c = Counter()
    for P in data.values():
        for r in P:
            if r["level"] == lv and r["fallacy"] in UNP:
                t, _ = an.first_collapse(r)
                if t: c[t] += 1
    n = sum(c.values()); dist[lv] = [c[t] / n for t in range(1, 6)]
fig, ax = plt.subplots(figsize=(HALF, 2.2))
bw = 0.26; xs = list(range(1, 6))
for i, lv in enumerate(LV3):
    off = (i - 1) * bw
    ax.bar([t + off for t in xs], dist[lv], width=bw - 0.03, color=SEQ[lv], edgecolor="white",
           linewidth=0.8, label=lv)
    ax.text(1 + off, dist[lv][0] + 0.015, f"{dist[lv][0]:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(xs); ax.set_xticklabels(["t1", "t2", "t3", "t4", "t5"])
ax.set_ylabel("Share of first collapses"); ax.set_ylim(0, 0.76); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.annotate("", xy=(1.6, 0.68), xytext=(5.4, 0.68), arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
ax.text(3.5, 0.69, "fallacy pressure", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.text(1, 0.69, "bare\nclaim", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.02, 0.9), handlelength=1.3)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_turn.pdf"); plt.close(fig)

print("turn-1 shares:", {lv: round(dist[lv][0], 2) for lv in LV3})
print("form:", {FLAB[k]: [round(v, 2) for v in vs] for k, vs in form.items()})
if "--png" in sys.argv:
    import pymupdf
    for n in ["fig_collapse", "fig_form", "fig_turn"]:
        pymupdf.open(f"figures/{n}.pdf")[0].get_pixmap(dpi=200).save(f"/tmp/tr/{n}.png")

# Fig 4: collapse rate by fallacy type, pooled over targets
FAL = [("appeal_to_authority", "appeal to authority", BLUE, "o", "-"),
       ("hasty_generalization", "hasty generalisation", ORANGE, "s", "-"),
       ("circular_reasoning", "circular reasoning", AQUA, "^", "-"),
       ("equivocation", "equivocation (manipulation check)", "#8a8884", "D", ":")]
allP = [r for P in data.values() for r in P]
fig, ax = plt.subplots(figsize=(W, 1.85))
for f, lab, col, mk, ls in FAL:
    ys, lo, hi = [], [], []
    for lv in LEVELS:
        k, n, _ = an.cell(allP, lv, {f})
        ys.append(k / n); a, b = an.wilson(k, n); lo.append(a); hi.append(b)
    ax.fill_between(x, lo, hi, color=col, alpha=0.08, linewidth=0)
    ax.plot(x, ys, lw=1.6, ms=4, color=col, marker=mk, ls=ls, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["L0", "L1", "L2", "L3", "L4", "L4P"])
ax.set_ylim(0, 1.02); ax.set_ylabel("Collapse rate"); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62), handlelength=2.4)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_fallacy.pdf"); plt.close(fig)
if "--png" in sys.argv:
    import pymupdf
    pymupdf.open("figures/fig_fallacy.pdf")[0].get_pixmap(dpi=200).save("/tmp/tr/fig_fallacy.png")
