"""Run one challenge episode end to end and write predictions.

    python run_episode.py --episode path/to/episode [--target-room A|B] [--out predictions.csv]

Episode layout (any of the source formats may be mixed):

    episode/
      query/                     captures to classify: *.pcap | *.csv (256 amplitude
                                 columns) | *.npy ((150, 256) amplitude windows)
      support/<ACTIVITY>/...     optional labelled target-room captures, one sub-folder
                                 per activity (e.g. support/FALL/x.pcap, support/EMPTY/y.csv)
      episode.yaml               optional: {target_room: A|B}

If there is no query/ folder, every input file outside support/ is a query.
Without support/, the target room's K=20 development support (seed 0) is used.
Without a target room, the router's majority vote over the query windows decides.

Output CSV columns: window_id, room, label, confidence (+ source_file).
Each query window is also saved as window_XXXX.npy next to the CSV.

Requires the trained artifacts from `python run_all.py` (normaliser, router,
SSL encoder, development embedding bank).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csihar.config import EXCLUSIVE, ROOM_ACTIVITIES, ROOMS, load_config, resolve  # noqa: E402
from csihar.dataset import load_split  # noqa: E402
from csihar.features import handcrafted  # noqa: E402
from csihar.fewshot import (BiasedClassifier, EnsembleClassifier, apply_alignment,  # noqa: E402
                            class_bias, fit_alignment, predict_restricted, refine_prototypes,
                            relabel_shared, select_support)
from csihar.inference import canonical_label, capture_windows, find_inputs, load_capture  # noqa: E402
from csihar.normalization import Normalizer  # noqa: E402
from csihar.pipeline import artifacts_dir, get_encoder  # noqa: E402
from csihar.room import RoomRouter  # noqa: E402
from csihar.ssl import embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--target-room", choices=ROOMS)
    ap.add_argument("--out")
    ap.add_argument("--config")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-relabel", action="store_true", help="keep the classifier's shared labels")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ep = Path(args.episode)
    out = Path(args.out) if args.out else ep / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_file = ep / "episode.yaml"
    ep_meta = yaml.safe_load(meta_file.read_text()) if meta_file.exists() else {}
    art = artifacts_dir(cfg)

    norm = Normalizer.load(art / "normalizer.json")
    router = RoomRouter.load(art / "router.pkl")
    enc = get_encoder(cfg)

    # ---- query windows (validation windowing policy)
    qdir = ep / "query" if (ep / "query").is_dir() else ep
    q_files = [f for f in find_inputs(qdir) if "support" not in f.relative_to(ep).parts]
    if not q_files:
        sys.exit(f"no query inputs found under {qdir}")
    windows, sources, skipped = [], [], []
    for f in q_files:
        amp = load_capture(f, cfg)
        if amp.shape[0] < cfg["window"]["min_valid"]:
            skipped.append((f, amp.shape[0]))
            continue
        for w in capture_windows(amp, cfg, "validation"):
            windows.append(w)
            sources.append(str(f.relative_to(ep)))
    for f, n in skipped:
        print(f"skipped {f}: only {n} valid observations after QC")
    Xq = np.stack(windows)
    Zq = norm.transform(Xq, scope="global")
    routed, room_conf = router.predict(Zq)
    Eq = embed(enc, Zq)

    target = args.target_room or ep_meta.get("target_room")
    if target is None:
        vals, cnt = np.unique(routed, return_counts=True)
        target = str(vals[cnt.argmax()])
    source = [r for r in ROOMS if r != target][0]
    print(f"{len(Xq)} query windows from {len(q_files) - len(skipped)} captures; target room {target}")

    # ---- development embedding bank (labels from the source room only)
    _, md = load_split(cfg, "development")
    Ed = np.load(art / "repr_ssl.npz")["dev"]
    rooms_dev = md.room.values
    stats = fit_alignment(Ed, rooms_dev, cfg["task3"]["alignment"])
    A_dev = apply_alignment(Ed, rooms_dev, stats)
    src = rooms_dev == source
    train_E = [A_dev[src]]
    train_y = [md.label.values[src]]
    train_g = [md.capture_id.values[src]]

    # ---- target support: from the episode, else default K=20 development support
    sdir = ep / "support"
    if sdir.is_dir():
        sup_E, sup_y = [], []
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            label = canonical_label(cls_dir.name)
            for f in find_inputs(cls_dir):
                amp = load_capture(f, cfg)
                if amp.shape[0] < cfg["window"]["min_valid"]:
                    continue
                Zs = norm.transform(np.stack(capture_windows(amp, cfg, "development")), scope="global")
                Es = apply_alignment(embed(enc, Zs), np.full(len(Zs), target), stats)
                train_E.append(Es)
                train_y.append(np.full(len(Es), label))
                train_g.append(np.full(len(Es), f"support:{f.name}"))
                sup_E.append(Es)
                sup_y.append(np.full(len(Es), label))
        sup_E, sup_y = np.concatenate(sup_E), np.concatenate(sup_y)
        unl = rooms_dev == target
        print(f"episode support: {dict(zip(*np.unique(sup_y, return_counts=True)))} windows")
    else:
        caps = select_support(md, target, 20, 0)
        sup = md.capture_id.isin(caps).values
        train_E.append(A_dev[sup])
        train_y.append(md.label.values[sup])
        train_g.append(md.capture_id.values[sup])
        sup_E, sup_y = A_dev[sup], md.label.values[sup]
        unl = (rooms_dev == target) & ~sup
        print("no episode support given: using K=20 development support captures per exclusive activity")

    clf = EnsembleClassifier().fit(np.concatenate(train_E), np.concatenate(train_y), np.concatenate(train_g))
    missing = set(EXCLUSIVE[target]) - set(clf.classes_)
    if missing:
        print(f"warning: no support for {sorted(missing)}; those activities cannot be predicted")
    if not args.no_refine:
        clf = refine_prototypes(clf, A_dev[unl], ROOM_ACTIVITIES[target], sup_E, sup_y)
    clf = BiasedClassifier(clf, class_bias(clf, A_dev[unl], ROOM_ACTIVITIES[target]))

    # ---- per-window routing: alignment stats and label space follow the routed room
    Aq = apply_alignment(Eq, routed, stats)
    allowed = [[c for c in ROOM_ACTIVITIES[r] if c in clf.classes_] for r in routed]
    labels, conf = predict_restricted(clf, Aq, allowed)

    # ---- shared SIT/STAND/WALK re-assigned from target-room clusters (motion-ranked)
    if not args.no_relabel:
        Hd = np.load(art / "repr_handcrafted.npz")["dev"]
        Md = np.load(art / "repr_motion.npz")["dev"]
        pred_unl, _ = predict_restricted(clf, A_dev[unl], [ROOM_ACTIVITIES[target]] * unl.sum())
        inroom = routed == target
        labels = labels.astype(object)
        F_unl = np.hstack([Hd[unl], A_dev[unl]])  # handcrafted + aligned SSL embedding
        F_q = np.hstack([handcrafted(Zq[inroom]), Aq[inroom]])
        labels[inroom] = relabel_shared(pred_unl, labels[inroom], F_unl, F_q,
                                        Md[unl, 0], k=cfg["task3"]["relabel_k"])

    win_dir = out.parent / (out.stem + "_windows")
    win_dir.mkdir(exist_ok=True)
    ids = []
    for i, w in enumerate(Xq, 1):
        wid = f"window_{i:04d}"
        np.save(win_dir / f"{wid}.npy", w.astype(np.float32))
        ids.append(wid)
    pd.DataFrame(dict(window_id=ids, room=routed, label=labels.astype(str), confidence=np.round(conf, 4),
                      room_confidence=np.round(room_conf, 4), source_file=sources)).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
