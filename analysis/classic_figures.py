"""Study design diagram and confusion matrices. All numbers come from common.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon

import common as fd

CLASSIC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 9.0,
    "axes.titlesize": 10.6, "axes.titleweight": "bold", "axes.titlepad": 12.0, "axes.titlelocation": "center",
    "axes.labelsize": 9.2, "axes.labelweight": "bold", "axes.labelcolor": "#1E242B",
    "axes.edgecolor": "#CFD6DE", "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.6, "xtick.color": "#4A5568", "ytick.color": "#1E242B",
    "xtick.labelcolor": "#4A5568", "ytick.labelcolor": "#1E242B", "text.color": "#1E242B",
    "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
}
C_TEXT = "#1E242B"
BLUES = LinearSegmentedColormap.from_list(
    "classic_modern_blues", ["#F8FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C", "#08306B"], N=256)


def num(n):
    n = int(n)
    return f"{n:,}" if n >= 10000 else str(n)


def fig_design(out):
    sp = fd.splits()
    lk = pd.read_csv(fd.CACHE / "resources" / "leakage_by_class.csv")
    p1_leak = lk[(lk.protocol == "p1_image_level") & (lk.partition == "test") & (lk["class"] == "ALL")].pct_leaked.iloc[0]
    n_test = int((sp.p1_image_level == "test").sum())
    n_off = len(fd.official_scored())
    n_models = len(fd.MAIN_MODELS)
    epochs = {len(pd.read_csv(fd.RUNS / fd.PROTOCOL[p] / r / s / "history.csv"))
              for p in fd.PROTOCOL for r, _ in fd.MAIN_MODELS for s in fd.seeds(r, p)
              if (fd.RUNS / fd.PROTOCOL[p] / r / s / "history.csv").exists()}
    assert len(epochs) == 1, epochs
    epochs = epochs.pop()

    with plt.rc_context(CLASSIC):
        figsize = (7.15, 3.4)
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_xlim(0.0, 16.4)
        ax.set_ylim(-2.65, 2.65)
        ax.set_aspect("equal")
        ax.axis("off")
        bw, bh, rad = 2.9, 1.6, 0.16
        fs_title, fs_sub = 8.6, 7.4

        def node(cx, cy, title, subtitle, fc, ec):
            ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                        boxstyle=f"round,pad=0.0,rounding_size={rad}",
                                        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
            ax.text(cx, cy + 0.33, title, ha="center", va="center", fontsize=fs_title, fontweight="bold",
                    color="#182026", linespacing=1.18, zorder=4)
            ax.text(cx, cy - 0.43, subtitle, ha="center", va="center", fontsize=fs_sub, fontweight="bold", color="#3E4C59",
                    linespacing=1.15, zorder=4)

        def ortho(pts, r=0.18, lw=1.5, col="#1E242B", arrow=True):
            pts = [np.array(p, dtype=float) for p in pts]
            head_l, head_w = 0.26, 0.14
            end = pts[-1].copy()
            u_last = (end - pts[-2]) / np.linalg.norm(end - pts[-2])
            if arrow:
                pts[-1] = end - u_last * head_l * 0.75
            verts, codes = [pts[0]], [MplPath.MOVETO]
            for i in range(1, len(pts) - 1):
                v_in, v_out = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
                rc = min(r, np.linalg.norm(v_in) * 0.45, np.linalg.norm(v_out) * 0.45)
                verts += [pts[i] - rc * v_in / np.linalg.norm(v_in), pts[i], pts[i] + rc * v_out / np.linalg.norm(v_out)]
                codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
            verts.append(pts[-1])
            codes.append(MplPath.LINETO)
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=col, linewidth=lw, zorder=2))
            if arrow:
                perp = np.array([-u_last[1], u_last[0]])
                base = end - u_last * head_l
                ax.add_patch(Polygon([end, base + perp * head_w, base - perp * head_w], closed=True,
                                     facecolor=col, edgecolor=col, linewidth=0.5, zorder=5))

        x1, x2, x3, x4, x5 = 1.45, 4.90, 8.20, 11.55, 14.90
        top, mid, bot = 1.8, 0.0, -1.8
        teal, p1, p2 = ("#D2E7E1", "#4E8275"), ("#F5DAD6", "#A13B35"), ("#D6E4F5", "#345E8E")
        gray, off, ev = ("#E2E5E9", "#64707D"), ("#EFE6D5", "#8A6D3B"), ("#DCE2E9", "#273A4B")

        node(x1, mid, "ISIC 2019\nTraining Archive", f"{num(len(sp))} Images\n{len(fd.CLASSES)} Classes", *teal)
        node(x2, top, "P1 Image-Level\nPartition", f"70 / 10 / 20\n{p1_leak:.1f}% Leaked", *p1)
        node(x2, bot, "P2 Lesion-Level\nPartition", "70 / 10 / 20\n0.0% Leaked", *p2)
        node(x3, top, "P1 Model\nTraining", f"{n_models} Networks\n{epochs} Epochs", *gray)
        node(x3, bot, "P2 Model\nTraining", f"{n_models} Networks\n{epochs} Epochs", *gray)
        node(x4, top, "P1 Internal\nEvaluation", f"{num(n_test)} Test Images", *p1)
        node(x4, mid, "Official Test\nEvaluation", f"{num(n_off)} Held-Out Images", *off)
        node(x4, bot, "P2 Internal\nEvaluation", f"{num(n_test)} Test Images", *p2)
        node(x5, mid, "Paired Benchmark\nContrasts", "Inflation · Gap · DiD", *ev)

        e = bw / 2
        fork = (x1 + e + x2 - e) / 2
        ortho([(x1 + e, mid), (fork, mid)], arrow=False)
        ortho([(fork, mid), (fork, top), (x2 - e, top)])
        ortho([(fork, mid), (fork, bot), (x2 - e, bot)])
        for yy in (top, bot):
            ortho([(x2 + e, yy), (x3 - e, yy)])
            ortho([(x3 + e, yy), (x4 - e, yy)])
        ortho([(x3, top - bh / 2), (x3, mid + 0.3), (x4 - e, mid + 0.3)])
        ortho([(x3, bot + bh / 2), (x3, mid - 0.3), (x4 - e, mid - 0.3)])
        ortho([(x4 + e, top), (x5, top), (x5, mid + bh / 2)])
        ortho([(x4 + e, mid), (x5 - e, mid)])
        ortho([(x4 + e, bot), (x5, bot), (x5, mid - bh / 2)])
        fig.savefig(out / "fig1_design.pdf", bbox_inches="tight", pad_inches=0.14,
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    print("wrote fig1_design (classic)")


def fig_confusion(out, run="B3_ConvNeXtV2T_GLC"):
    panels = [(0, 0, "P1", "test", "(a) P1 Trained Internal Test"), (0, 1, "P1", "official", "(b) P1 Trained Official Test"),
              (1, 0, "P2", "test", "(c) P2 Trained Internal Test"), (1, 1, "P2", "official", "(d) P2 Trained Official Test")]
    with plt.rc_context(CLASSIC):
        fig = plt.figure(figsize=(7.0, 6.6))
        gs = fig.add_gridspec(2, 2, left=0.09, right=0.88, bottom=0.07, top=0.95, hspace=0.34, wspace=0.08)
        axes, im = {}, None
        for r, c, prot, ts, title in panels:
            mat = fd.confusion(run, prot, ts)
            ax = fig.add_subplot(gs[r, c])
            axes[(r, c)] = ax
            im = ax.imshow(mat, cmap=BLUES, vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(np.arange(8), fd.CLASSES, fontsize=7.6, fontweight="bold")
            ax.set_yticks(np.arange(8), fd.CLASSES, fontsize=7.6, fontweight="bold")
            ax.tick_params(which="major", length=0, pad=5)
            if c == 0:
                ax.set_ylabel("True Diagnosis", fontsize=8.2, fontweight="bold", labelpad=5)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel("Predicted Diagnosis", fontsize=8.2, fontweight="bold", labelpad=4)
            ax.set_title(title, loc="center", fontsize=9.0, fontweight="bold", color=C_TEXT, pad=10)
            for i in range(8):
                for j in range(8):
                    v = round(mat[i, j], 2)
                    if v > 0.50:
                        col, fw = "white", "bold"
                    elif i == j:
                        col, fw = "#08306B", "bold"
                    elif v < 0.005:
                        col, fw = "#7A8794", "bold"
                    else:
                        col, fw = "#1E293B", "bold"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2, fontweight=fw, color=col)
        fig.canvas.draw()
        pt, pb = axes[(0, 1)].get_position(), axes[(1, 1)].get_position()
        cax = fig.add_axes([pt.x1 + 0.02, pb.y0, 0.016, pt.y1 - pb.y0])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.set_label("Row-Normalised Recall", fontsize=8.2, fontweight="bold", color=C_TEXT, labelpad=9)
        cb.ax.tick_params(labelsize=7.6, width=0.6, length=2.5, color="#8C98A4")
        fig.savefig(out / "fig_confusion.pdf", bbox_inches="tight", pad_inches=0.14,
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    print("wrote fig_confusion (classic)")
