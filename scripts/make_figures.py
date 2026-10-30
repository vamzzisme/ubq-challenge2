"""Required figures (all from VALIDATION results written by the task scripts).

  fig_label_efficiency.png   macro-F1 vs K per target room; target-labelled
                             (exclusive) vs transferred shared activities
  fig_embedding_2d.png       2-D embeddings coloured by room and by activity
  fig_room_vs_length.png     room accuracy vs window length
  fig_routing.png            predicted routing vs oracle room vs no routing
  fig_confusion_K5.png       confusion matrices at K=5 with P/R/F1
  fig_ablation.png           representation / refinement ablation
  fig_ssl_loss.png           SSL training curve
  fig_qc_examples.png        decoded Room A amplitude (valid vs legacy) and Room B
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import load_config, resolve  # noqa: E402

PALETTE = ["#2a6fdb", "#e0782f", "#2a9d6f", "#c43c5c", "#7b5cd6", "#8c8c8c", "#b5942a", "#3fb4c9"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def label_efficiency(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, target in zip(axes, ("A", "B")):
        s = summary[(summary.experiment == "task3_main") & (summary.target_room == target)]
        for (metric, name), c in zip([("macro_f1", "all activities (primary)"),
                                      ("target_exclusive_macro_f1", "target-labelled (exclusive)"),
                                      ("shared_macro_f1", "transferred shared (SIT/STAND/WALK)")], PALETTE):
            d = s[s.metric == metric].sort_values("K")
            ax.errorbar(d.K, d["mean"], yerr=d["std"], marker="o", capsize=3, color=c, label=name)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K (labelled target captures per exclusive activity)")
        ax.set_title(f"Target room {target} ({'B→A' if target == 'A' else 'A→B'})")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[1].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def embedding(df, out):
    kinds = list(df.kind.unique())
    fig, axes = plt.subplots(len(kinds), 2, figsize=(9, 4 * len(kinds)))
    axes = np.atleast_2d(axes)
    for i, kind in enumerate(kinds):
        d = df[df.kind == kind]
        for j, col in enumerate(("room", "label")):
            ax = axes[i, j]
            for k, (val, g) in enumerate(d.groupby(col)):
                ax.scatter(g.x, g.y, s=9, color=PALETTE[k % len(PALETTE)], label=val, alpha=0.8, lw=0)
            ax.set_title(f"{kind} encoder ({d.method.iloc[0].upper()}) - coloured by {col}")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=7, markerscale=1.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def room_vs_length(df, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(df.window_length, df.accuracy, marker="o", color=PALETTE[0], label="overall")
    ax.plot(df.window_length, df.acc_A, marker=".", ls="--", color=PALETTE[1], label="Room A windows")
    ax.plot(df.window_length, df.acc_B, marker=".", ls="--", color=PALETTE[2], label="Room B windows")
    ax.set_xscale("log")
    ax.set_xlabel("window length (observations)")
    ax.set_ylabel("room accuracy (validation)")
    ax.set_ylim(min(0.5, df[["accuracy", "acc_A", "acc_B"]].values.min() - 0.02), 1.01)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def routing(summary, out):
    names = [("task3_main", "predicted-room routing"), ("task3_oracle_room", "oracle room"),
             ("task3_no_routing", "no routing")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, target in zip(axes, ("A", "B")):
        w = 0.26
        for i, ((exp, name), c) in enumerate(zip(names, PALETTE)):
            d = summary[(summary.experiment == exp) & (summary.target_room == target)
                        & (summary.metric == "macro_f1")].sort_values("K")
            x = np.arange(len(d)) + (i - 1) * w
            ax.bar(x, d["mean"], w, yerr=d["std"], color=c, label=name, capsize=2)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels([f"K={k}" for k in d.K])
        ax.set_title(f"Target room {target}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def confusion(res_dir, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, target in zip(axes, ("A", "B")):
        df = pd.read_csv(res_dir / f"task3_confusion_K5_target{target}.csv", index_col=0)
        labels = list(df.index)
        cm = df[labels].values
        ax.imshow(cm, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f"{l}  P={df.precision[l]:.2f} R={df.recall[l]:.2f} F1={df.f1[l]:.2f}"
                            for l in labels], fontsize=7.5)
        ax.set_xlabel("predicted")
        ax.set_title(f"Target room {target}, K=5 (seed 0)")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def ablation(summary, out):
    exps = [("task3_main", "SSL, ensemble + shared relabel (main)"),
            ("task3_no_relabel", "SSL, ensemble + calibration (no relabel)"),
            ("task3_proto_baseline", "SSL, prototypes (first main)"),
            ("task3_relabel_handcrafted", "relabel on handcrafted only"),
            ("task3_relabel_motion", "relabel on motion features"),
            ("task3_logreg", "SSL + logreg"), ("task3_random_encoder", "random encoder"),
            ("task3_handcrafted", "handcrafted")]
    exps = [e for e in exps if e[0] in set(summary.experiment)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, target in zip(axes, ("A", "B")):
        for (exp, name), c in zip(exps, PALETTE):
            d = summary[(summary.experiment == exp) & (summary.target_room == target)
                        & (summary.metric == "macro_f1")].sort_values("K")
            ax.plot(d.K, d["mean"], marker="o", color=c, label=name)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_title(f"Target room {target}: primary macro-F1")
        ax.set_xlabel("K")
        ax.set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def ssl_loss(path, out):
    hist = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.plot(np.arange(1, len(hist) + 1), hist, color=PALETTE[0])
    ax.set_xlabel("epoch")
    ax.set_ylabel("NT-Xent loss")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def qc_examples(cfg, out):
    from csihar.cleaning import qc_room_a
    from csihar.csi_decode import decode_payloads
    from csihar.pcap_reader import read_udp_payloads
    exp2 = resolve(cfg, "exp2_dir")
    f = sorted((exp2 / "walk").glob("*.pcap"))[0]
    p, t, _ = read_udp_payloads(f)
    fr = decode_payloads(p, t)
    keep, _, _ = qc_room_a(fr)
    amp = np.abs(fr.csi)
    bfile = sorted(resolve(cfg, "exp3_dir").glob("Walking_*.csv"))[0]
    b = pd.read_csv(bfile).to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.arange(-128, 128)
    axes[0].plot(x, np.fft.fftshift(amp[keep].mean(0)), color=PALETTE[0], label="valid VHT80 frames")
    axes[0].plot(x, np.fft.fftshift(amp[~keep].mean(0)), color=PALETTE[1], label="rejected legacy-20 MHz")
    axes[0].set_title(f"Room A {f.name}: mean |H|")
    axes[0].set_xlabel("subcarrier index (centred)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].imshow(amp[keep].T, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Room A valid frames (time x FFT bin)")
    axes[1].set_xlabel("valid observation")
    axes[1].grid(False)
    axes[2].imshow(b[:150].T, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title(f"Room B {bfile.name} (first 150 rows)")
    axes[2].set_xlabel("row")
    axes[2].grid(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    cfg = load_config()
    res = resolve(cfg, "results")
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(res / "task3_summary.csv")
    label_efficiency(summary, figs / "fig_label_efficiency.png")
    routing(summary, figs / "fig_routing.png")
    ablation(summary, figs / "fig_ablation.png")
    confusion(res, figs / "fig_confusion_K5.png")
    embedding(pd.read_csv(res / "task2_embedding_2d.csv"), figs / "fig_embedding_2d.png")
    room_vs_length(pd.read_csv(res / "task1_window_length.csv"), figs / "fig_room_vs_length.png")
    hist = resolve(cfg, "artifacts") / "ssl_history.json"
    if hist.exists():
        ssl_loss(hist, figs / "fig_ssl_loss.png")
    qc_examples(cfg, figs / "fig_qc_examples.png")
    print("figures written to", figs)


if __name__ == "__main__":
    main()
