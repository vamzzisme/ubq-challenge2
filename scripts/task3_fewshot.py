"""Task 3: cross-room few-label adaptation, both directions, K in {1,5,10,20}.

For every (representation, method, routing, target room, K, support seed)
an episode is run (see csihar.fewshot). Every number is a VALIDATION result.

Outputs
  results/task3_results.csv      per-episode metrics (experiment,target_room,K,metric,value,seed)
  results/task3_summary.csv      mean/std over support seeds
  results/task3_predictions_*.csv  window_id,room,label,confidence (main method)
  results/task3_confusion_K5_*.csv  confusion matrices at K=5 (main method, seed 0)
"""
import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import EXCLUSIVE, ROOM_ACTIVITIES, SHARED, load_config, resolve  # noqa: E402
from csihar.fewshot import fit_alignment, run_episode  # noqa: E402
from csihar.metrics import ResultLog, balanced_acc, macro_f1  # noqa: E402
from csihar.pipeline import artifacts_dir, load_normalized, representations  # noqa: E402
from csihar.room import RoomRouter  # noqa: E402

# scoring restricted label sets: out-of-room predictions are simply errors
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# (name, representation, classifier, refine, routing, alignment, calibrate, relabel);
# alignment None means the configured default (task3.alignment). relabel None
# keeps the classifier's shared labels; otherwise it names the feature space whose
# target-room clusters re-assign them: "handcrafted+emb" (handcrafted features plus
# the variant's aligned embedding), "handcrafted" or "motion".
VARIANTS = [
    ("main", "ssl", "ensemble", True, "predicted", None, True, "handcrafted+emb"),
    ("no_relabel", "ssl", "ensemble", True, "predicted", None, True, None),  # second main method
    ("relabel_handcrafted", "ssl", "ensemble", True, "predicted", None, True, "handcrafted"),
    ("proto_baseline", "ssl", "proto", True, "predicted", None, False, None),  # first main method
    ("relabel_motion", "ssl", "ensemble", True, "predicted", None, True, "motion"),
    ("oracle_room", "ssl", "ensemble", True, "oracle", None, True, "handcrafted+emb"),
    ("no_routing", "ssl", "ensemble", True, "none", None, True, "handcrafted+emb"),
    ("no_refine", "ssl", "ensemble", False, "predicted", None, True, "handcrafted+emb"),
    ("no_calibrate", "ssl", "ensemble", True, "predicted", None, False, "handcrafted+emb"),
    ("diag_alignment", "ssl", "ensemble", True, "predicted", "diag", True, "handcrafted+emb"),
    ("logreg", "ssl", "logreg", False, "predicted", None, False, None),
    ("random_encoder", "random", "ensemble", True, "predicted", None, True, "handcrafted+emb"),
    ("random_encoder_no_refine", "random", "ensemble", False, "predicted", None, True, "handcrafted+emb"),
    ("handcrafted", "handcrafted", "ensemble", True, "predicted", None, True, "handcrafted+emb"),
    ("handcrafted_no_refine", "handcrafted", "ensemble", False, "predicted", None, True, "handcrafted+emb"),
    ("ssl_dyn", "ssl_dyn", "ensemble", True, "predicted", None, True, "handcrafted+emb"),  # static profile removed
]


def score(r, target):
    y, p = r["y_true"], r["y_pred"]
    return {
        "macro_f1": macro_f1(y, p, ROOM_ACTIVITIES[target]),
        "target_exclusive_macro_f1": macro_f1(y, p, EXCLUSIVE[target]),
        "shared_macro_f1": macro_f1(y, p, SHARED),
        "balanced_accuracy": balanced_acc(y, p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--variants", help="comma-separated subset of variant names")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = resolve(cfg, "results")
    log = ResultLog(res_dir / "task3_results.csv")

    variants = VARIANTS
    if args.variants:
        keep = set(args.variants.split(","))
        variants = [v for v in VARIANTS if v[0] in keep]

    Zd, md, Zv, mv, _ = load_normalized(cfg)
    reps = representations(cfg, Zd, Zv, md, kinds=sorted({v[1] for v in variants} | {"handcrafted", "motion"}))
    motion_dev = reps["motion"][0][:, 0]  # slow-motion score that names the shared clusters
    router = RoomRouter.load(artifacts_dir(cfg) / "router.pkl")
    routed, _ = router.predict(Zv)
    print("router accuracy on validation:", np.mean(routed == mv.room.values))

    Ks, seeds = cfg["task3"]["K"], cfg["task3"]["support_seeds"]
    cm_rows = []  # K=5 confusion counts summed over support seeds, every variant
    for name, rep, method, refine, routing, alignment, calibrate, relabel_space in variants:
        Ed, Ev = reps[rep]
        relabel = None
        if relabel_space is not None:
            space = relabel_space.removesuffix("+emb")
            relabel = (*reps[space], motion_dev, cfg["task3"]["relabel_k"], relabel_space.endswith("+emb"))
        stats = fit_alignment(Ed, md.room.values, alignment or cfg["task3"]["alignment"])
        for target, K, seed in itertools.product(("A", "B"), Ks, seeds):
            r = run_episode(Ed, md, Ev, mv, target, K, seed, routed, stats,
                            classifier=method, routing=routing, refine=refine, calibrate=calibrate,
                            relabel=relabel)
            direction = "B->A" if target == "A" else "A->B"
            for metric, value in score(r, target).items():
                log.add(f"task3_{name}", metric, value, target_room=target, K=K, seed=seed, notes=direction)
            if K == 5:
                cm_rows += [dict(experiment=f"task3_{name}", target_room=target, seed=seed, true=t, pred=p)
                            for t, p in zip(r["y_true"], r["y_pred"])]
            if name == "main" and seed == seeds[0]:
                pd.DataFrame(dict(window_id=r["window_id"], room=r["routed_room"], label=r["y_pred"],
                                  confidence=np.round(r["confidence"], 4), true_label=r["y_true"])) \
                    .to_csv(res_dir / f"task3_predictions_target{target}_K{K}.csv", index=False)
                if K == 5:
                    labels = ROOM_ACTIVITIES[target]
                    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)
                    pr, rc, f1, sup = precision_recall_fscore_support(
                        r["y_true"], r["y_pred"], labels=labels, zero_division=0)
                    df = pd.DataFrame(cm, index=labels, columns=labels)
                    df["precision"], df["recall"], df["f1"], df["support"] = pr, rc, f1, sup
                    df.to_csv(res_dir / f"task3_confusion_K5_target{target}.csv")
        m = pd.DataFrame(log.rows)
        m = m[(m.experiment == f"task3_{name}") & (m.metric == "macro_f1")]
        print(name, m.groupby(["target_room", "K"]).value.mean().round(3).to_dict(), flush=True)
    log.save()
    pd.DataFrame(cm_rows).groupby(["experiment", "target_room", "true", "pred"]).size() \
        .rename("count").reset_index().to_csv(res_dir / "task3_confusion_K5_variants.csv", index=False)

    df = pd.DataFrame(log.rows)
    summary = df.groupby(["experiment", "target_room", "K", "metric"]).value.agg(["mean", "std"]).reset_index()
    summary.to_csv(res_dir / "task3_summary.csv", index=False)


if __name__ == "__main__":
    main()
