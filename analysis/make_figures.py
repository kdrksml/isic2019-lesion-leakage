"""Draw the figures of the paper from the released predictions.

Every plotted number is computed in common.py from the per-image prediction files, the split table or the leakage
audit. Nothing is typed in by hand. Figures are drawn close to their printed width, so the font sizes below are
close to the sizes on the page. Run: python make_figures.py  (writes into ../figures)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import classic_figures
import common as fd

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

INK, CLAY = "#30558C", "#C56A3E"
TEXT, MUTED, SPINE, GRID, REF = "#1F2328", "#4B5563", "#3A3F47", "#E4E7EB", "#8A9099"
TIDE = ["#1B2A41", "#34496A", "#5B7697", "#8EA3BC", "#D4A86A", "#C56A3E", "#A2412B", "#7E1F23"]
TEXTWIDTH = 13.8 / 2.54   # text width of the journal layout, inches
WIDE = TEXTWIDTH          # figures sit in the text column
CLASS_NAME = {"MEL": "Melanoma", "NV": "Melanocytic nevus", "BCC": "Basal cell carcinoma",
              "AK": "Actinic keratosis", "BKL": "Benign keratosis", "DF": "Dermatofibroma",
              "VASC": "Vascular lesion", "SCC": "Squamous cell carcinoma"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "mathtext.fontset": "custom", "mathtext.rm": "Arial:bold", "mathtext.bf": "Arial:bold",
    "font.size": 9, "axes.labelsize": 9.5, "axes.labelweight": "bold", "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": SPINE, "ytick.color": SPINE,
    "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SPINE, "axes.linewidth": 0.9,
    "axes.labelpad": 4, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "legend.handlelength": 1.8,
    "axes.titlelocation": "left", "axes.titlepad": 8, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def num(n):
    """Journal style: no thousands separator up to four digits, comma from five digits."""
    n = int(n)
    return f"{n:,}" if n >= 10000 else str(n)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}")


def open_axes(ax, value_axis="y"):
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.grid(axis=value_axis, zorder=0)


def category_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig, name, raster=False):
    if raster:
        fig.savefig(OUT / f"{name}.png", dpi=400)
    else:
        fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- examples
def fig_examples():
    tiles = fd.CACHE / "resources" / "figure_tiles_256.npz"
    if not tiles.exists():
        print("skipped fig_examples: needs image tiles cut from the ISIC 2019 images")
        return
    npz = np.load(tiles)
    rows = [("MEL", "BCN20000"), ("NV", "HAM10000"), ("BCC", "BCN20000"), ("BKL", "HAM10000")]
    H = WIDE * 0.78
    fig = plt.figure(figsize=(WIDE, H))
    n, gap = 8, 0.008
    tw = (1.0 - (n - 1) * gap) / n
    th = tw * WIDE / H
    y = 1.0 - 0.085 - th
    fig.text(0.0, 1.0, "(a) One image per diagnostic class", va="top", fontsize=10)
    for i, cls in enumerate(fd.CLASSES):
        ax = fig.add_axes([i * (tw + gap), y, tw, th])
        ax.imshow(npz["top8_imgs"][i])
        ax.axis("off")
        fig.text(i * (tw + gap) + tw / 2, y + th + 0.008, cls, ha="center", va="bottom", fontsize=9.5)
    y -= 0.07
    fig.text(0.0, y + 0.03, "(b) Repeated photographs of one lesion", va="top", fontsize=10)
    for r, (cls, source) in enumerate(rows):
        imgs = npz[f"grp_{cls}"]
        yr = y - th - r * (th + 0.022)
        fig.text(tw - 0.01, yr + th / 2, f"{cls}\n{source}\n{len(imgs)} images", ha="right", va="center",
                 fontsize=9, linespacing=1.35)
        for k, img in enumerate(imgs):
            ax = fig.add_axes([(k + 1) * (tw + gap), yr, tw, th])
            ax.imshow(img)
            ax.axis("off")
    save(fig, "fig_examples", raster=True)


# ---------------------------------------------------------------- dataset
def fig_dataset():
    sp = fd.splits()
    counts = sp.label.value_counts()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.7))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, height=0.7, color=TIDE, zorder=2)
    ax.set_yticks(y, [f"{CLASS_NAME[c]} ({c})" for c in counts.index])
    for yi, v in zip(y, counts.values):
        ax.text(v + 180, yi, f"{num(v)} ({100 * v / len(sp):.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, 15500)
    ax.set_xticks(np.arange(0, 15000, 3000))
    ax.xaxis.set_major_formatter(lambda v, _: num(v))
    ax.set_xlabel("Training images")
    open_axes(ax, "x")
    category_axis(ax)
    save(fig, "fig_dataset")


# ---------------------------------------------------------------- leakage and its dose
P1C, P2C = TIDE[6], TIDE[1]


def rank_colours(n):
    """Warm to cool by rank, largest value first, as in the dataset figure."""
    return TIDE[::-1][:n] if n == len(TIDE) else [TIDE[::-1][round(k * (len(TIDE) - 1) / (n - 1))] for k in range(n)]


SMALL = {"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "legend.fontsize": 8}


@plt.rc_context(SMALL)
def fig_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.3), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.05]})

    ax = axes[0]
    lk = fd.leakage_by_class()
    lk_all = pd.read_csv(fd.CACHE / "resources" / "leakage_by_class.csv")
    overall = lk_all[(lk_all.protocol == "p1_image_level") & (lk_all.partition == "test")
                     & (lk_all["class"] == "ALL")].pct_leaked.iloc[0]
    y = np.arange(len(lk))[::-1]
    ax.axvline(overall, color=REF, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.text(overall, len(lk) - 0.3, f"all classes {overall:.1f}%", ha="center", va="bottom", fontsize=7.5,
            color=MUTED, zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.barh(y, lk.pct_leaked, height=0.66, color=rank_colours(len(lk)), zorder=2)
    for yi, v, n in zip(y, lk.pct_leaked, lk.n):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.text(119, yi, f"n = {num(n)}", ha="right", va="center", fontsize=7.5, color=MUTED, clip_on=False)
    ax.set_yticks(y, lk["class"])
    ax.set_ylim(-0.6, len(lk) + 0.45)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Leaked P1 test images, %")
    open_axes(ax, "x")
    category_axis(ax)
    panel(ax, "a", "Same-group leakage by class")

    ax = axes[1]
    dose, counts = dose_standardised()
    xs = np.arange(dose.shape[1])
    for _, row in dose.iterrows():
        ax.plot(xs, row.values, color=P1C, alpha=0.18, lw=0.9, zorder=2)
    mean = dose.mean().values
    ax.plot(xs, mean, color=P1C, lw=2.2, marker="o", ms=6.5, mfc=P1C, mec="white", mew=1.2, zorder=4,
            label="P1 models, mean of ten")
    for xi, v in zip(xs, mean):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))
    ax.axhline(p2_internal_accuracy(), color=P2C, ls=(0, (5, 3)), lw=1.3, zorder=3, label="P2 internal test")
    ax.set_xticks(xs, [f"{c}\n(n = {num(n)})" for c, n in zip(dose.columns, counts.values)])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(-0.4, xs[-1] + 0.4)
    ax.set_xlabel("Same-group images in P1 training")
    ax.set_ylabel("Class-standardised accuracy")
    ax.set_ylim(0.70, 1.0)
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    open_axes(ax)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    panel(ax, "b", "Accuracy by same-group count")
    save(fig, "fig_leakage")


def dose_standardised():
    """P1 test accuracy per same-lesion bin, standardised to the class mix of the P1 test partition."""
    n_same = fd.same_lesion_counts()
    labels = ["0", "1", "2–3", "≥4"]
    bins = pd.cut(n_same, [-1, 0, 1, 3, np.inf], labels=labels)
    rows = {}
    for run, label in fd.MAIN_MODELS:
        d = fd.predictions(run, "P1", "test").set_index("image")
        d["bin"] = bins.reindex(d.index)
        d["ok"] = d.y_pred == d.y_true
        w = d.y_true.value_counts(normalize=True)
        per = d.groupby(["bin", "y_true"], observed=True).ok.mean().unstack()
        rows[label] = (per * w).sum(1) / per.notna().mul(w).sum(1)
    return pd.DataFrame(rows).T[labels], bins.value_counts().reindex(labels)


def p2_internal_accuracy():
    ev = fd.evaluation()
    return float(np.mean([ev[(ev.protocol == "lesion") & (ev.name == r) & (ev.testset == "test")].accuracy.mean()
                          for r, _ in fd.MAIN_MODELS]))


# ---------------------------------------------------------------- common test images and per-class inflation
def shared_accuracy():
    sp = fd.splits()
    common = set(sp[(sp.p1_image_level == "test") & (sp.p2_lesion_level == "test")].image)
    leaked = common & set(fd.same_lesion_counts().loc[lambda s: s > 0].index)
    out = {"P1": {"leaked": [], "unleaked": []}, "P2": {"leaked": [], "unleaked": []},
           "n_leaked": len(leaked), "n_unleaked": len(common - leaked)}
    for run, _ in fd.MAIN_MODELS:
        for prot in ("P1", "P2"):
            d = fd.predictions(run, prot, "test")
            d = d[d.image.isin(common)]
            m = d.image.isin(leaked)
            out[prot]["leaked"].append(np.mean(d.y_pred[m] == d.y_true[m]))
            out[prot]["unleaked"].append(np.mean(d.y_pred[~m] == d.y_true[~m]))
    return out


@plt.rc_context(SMALL)
def fig_common():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.1]})
    whisker = dict(color=TEXT, lw=1.0, capsize=3, capthick=1.0, zorder=3)

    ax = axes[0]
    shared = shared_accuracy()
    width = 0.38
    for i, part in enumerate(("leaked", "unleaked")):
        for j, (prot, col) in enumerate((("P1", P1C), ("P2", P2C))):
            vals = np.array(shared[prot][part])
            xpos = i + (j - 0.5) * (width + 0.04)
            ax.bar(xpos, vals.mean(), width, color=col, zorder=2, label=f"{prot} models" if i == 0 else None)
            ax.errorbar(xpos, vals.mean(), yerr=[[vals.mean() - vals.min()], [vals.max() - vals.mean()]], **whisker)
            ax.text(xpos, vals.max() + 0.025, f"{vals.mean():.3f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1], [f"Leaked\n(n = {shared['n_leaked']})", f"Not leaked\n(n = {shared['n_unleaked']})"])
    ax.set_ylabel("Accuracy on the common images")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.6, 1.6)
    open_axes(ax)
    ax.spines["left"].set_bounds(0, 1)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.4)
    panel(ax, "a", "Test images common to both splits")

    ax = axes[1]
    infl = fd.class_f1_inflation()
    order = infl.mean().sort_values(ascending=False).index
    y = np.arange(len(order))[::-1]
    means = infl[order].mean().values
    ax.barh(y, means, height=0.66, color=rank_colours(len(order)), zorder=2)
    ax.errorbar(means, y, xerr=[means - infl[order].min().values, infl[order].max().values - means], fmt="none",
                **whisker)
    for yi, cls in zip(y, order):
        ax.text(infl[cls].max() + 0.012, yi, f"+{infl[cls].mean():.2f}", va="center", fontsize=8)
    ax.set_yticks(y, order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.41, 0.1))
    ax.set_xlabel("F1 inflation, P1 minus P2")
    open_axes(ax, "x")
    category_axis(ax)
    ax.spines["bottom"].set_bounds(0, 0.4)
    panel(ax, "b", "Per-class F1 inflation")
    save(fig, "fig_common")


# ---------------------------------------------------------------- internal vs official
@plt.rc_context(SMALL)
def fig_estimate_vs_official():
    tab = fd.macro_f1_table()
    short = {"SwinV2-T, shared recipe": "SwinV2-T", "ConvNeXtV2-T, shared recipe": "ConvNeXtV2-T",
             "SwinV2-T, shared recipe + GLC": "SwinV2-T + GLC", "ConvNeXtV2-T, shared recipe + GLC": "ConvNeXtV2-T + GLC"}
    groups = [("Plain recipe", tab.iloc[:6]), ("Shared recipe", tab.iloc[6:10]), ("Exploratory, 384 px", tab.iloc[10:])]
    p1c = CLAY
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    ypos, labels, y = [], [], 0.0
    for gi, (gname, g) in enumerate(groups):
        if gi:
            y -= 0.6
        ax.text(0.455, y + 0.05, gname, fontsize=9, fontstyle="italic", color=MUTED, va="bottom")
        y -= 0.75
        for _, r in g.iterrows():
            for prot, col, dy in (("P1", p1c, 0.18), ("P2", P2C, -0.18)):
                a, b = r[f"{prot}_test"], r[f"{prot}_official"]
                ax.annotate("", xy=(b, y + dy), xytext=(a, y + dy),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=col, lw=1.5,
                                            shrinkA=3.5, shrinkB=3.5))
                ax.plot(a, y + dy, "o", color=col, ms=6, zorder=3)
                ax.plot(b, y + dy, "o", mfc="white", mec=col, mew=1.5, ms=6, zorder=3)
            ypos.append(y)
            labels.append(short.get(r.label, r.label).replace(", 384 px", ""))
            y -= 1.0
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0.45, 0.90)
    ax.set_ylim(y + 0.45, 0.9)
    ax.set_xlabel("Macro-F1")
    open_axes(ax, "x")
    category_axis(ax)
    handles = [Line2D([], [], marker="o", ls="none", color=p1c, ms=6, label="P1 internal test"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=p1c, mew=1.5, ms=6, label="P1 official test"),
               Line2D([], [], marker="o", ls="none", color=P2C, ms=6, label="P2 internal test"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=P2C, mew=1.5, ms=6, label="P2 official test")]
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), columnspacing=1.6)
    save(fig, "fig3_estimate_vs_official")


if __name__ == "__main__":
    fig_examples()
    fig_dataset()
    classic_figures.fig_design(OUT)
    fig_leakage()
    fig_common()
    fig_estimate_vs_official()
    classic_figures.fig_confusion(OUT)
