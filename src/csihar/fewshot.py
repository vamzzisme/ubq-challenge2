"""Task 3: cross-room few-label adaptation on frozen representations.

Pipeline for one episode (target room T, source room S, K, seed):

  1. Support selection: for every T-exclusive activity, draw K distinct
     T-development captures (sorted capture ids, numpy RNG with `seed`).
     All windows of a chosen capture are used, but K counts captures.
  2. Environment-aware alignment: embeddings are whitened with the
     statistics of their room's *unlabelled* development embeddings --
     per-dimension mean/std (a feature-level analogue of AdaBN) or full
     covariance (CORAL-style). At inference the room comes from the
     router (Task 1 classifier), the oracle room, or -- without routing --
     one pooled set of statistics.
  3. Classifier on aligned embeddings, trained on all labelled S-development
     windows plus the T support windows:
       proto  - cosine nearest class prototype (per-capture means averaged,
                so one long capture does not dominate a class)
       logreg - class-balanced multinomial logistic regression
  4. Label-space routing: a window routed to room R may only receive R's
     activities. Without routing, the union of all trained labels is used.
  5. Optional transductive refinement on unlabelled T-development windows
     (never validation): soft prototype re-estimation that moves the
     source-learned shared-class prototypes toward the target room.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import EXCLUSIVE, ROOM_ACTIVITIES, ROOMS, SHARED


# ---------------------------------------------------------------- alignment
def _whitener(e, mode, shrink=0.1):
    mu = e.mean(0)
    if mode == "diag":
        return mu, np.diag(1.0 / (e.std(0) + 1e-6))
    # CORAL-style full whitening with shrinkage towards the diagonal
    c = np.cov(e, rowvar=False)
    c = (1 - shrink) * c + shrink * np.diag(np.diag(c))
    w, v = np.linalg.eigh(c)
    return mu, v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ v.T


def fit_alignment(E_dev, rooms_dev, mode="diag"):
    """Per-room (and pooled) whitening statistics from unlabelled dev embeddings.

    mode: 'diag' (per-dimension standardisation) or 'coral' (full covariance).
    """
    rooms_dev = np.asarray(rooms_dev)
    stats = {"global": _whitener(E_dev, mode)}
    for r in ROOMS:
        stats[r] = _whitener(E_dev[rooms_dev == r], mode)
    return stats


def apply_alignment(E, rooms, stats):
    """rooms: per-window room for room-wise stats, or None for pooled stats."""
    if rooms is None:
        mu, W = stats["global"]
        return ((E - mu) @ W).astype(np.float32)
    out = np.empty_like(E)
    rooms = np.asarray(rooms)
    for r in ROOMS:
        m = rooms == r
        mu, W = stats[r]
        out[m] = (E[m] - mu) @ W
    return out


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


# --------------------------------------------------------------- classifiers
class PrototypeClassifier:
    def __init__(self, temperature=0.1):
        self.t = temperature

    def fit(self, E, y, groups):
        self.classes_ = np.array(sorted(set(y)))
        En = _l2(E)
        protos = []
        for c in self.classes_:
            m = y == c
            caps = np.unique(groups[m])
            per_cap = np.stack([En[m & (groups == g)].mean(0) for g in caps])
            protos.append(per_cap.mean(0))
        self.protos_ = _l2(np.stack(protos))
        return self

    def decision(self, E):
        return _l2(E) @ self.protos_.T / self.t


class LogRegClassifier:
    def __init__(self, C=0.5):
        self.m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def fit(self, E, y, groups):
        self.m.fit(E, y)
        self.classes_ = self.m.classes_
        return self

    def decision(self, E):
        return self.m.decision_function(E)


def _softmax(s):
    p = np.exp(s - s.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


class EnsembleClassifier:
    """Average of prototype and logistic-regression class probabilities."""

    def __init__(self):
        self.parts = [PrototypeClassifier(), LogRegClassifier()]

    def fit(self, E, y, groups):
        for p in self.parts:
            p.fit(E, y, groups)
        self.classes_ = self.parts[0].classes_  # both sort labels the same way
        return self

    def decision(self, E):
        P = sum(_softmax(p.decision(E)) for p in self.parts) / len(self.parts)
        return np.log(np.maximum(P, 1e-12))


class BiasedClassifier:
    """Adds fixed per-class logit offsets to another classifier's scores."""

    def __init__(self, clf, bias):
        self.clf, self.bias, self.classes_ = clf, bias, clf.classes_

    def decision(self, E):
        return self.clf.decision(E) + self.bias


def class_bias(clf, E_unlab, allowed, n_iter=100):
    """Per-class logit offsets that make predictions on unlabelled target-dev
    windows class-balanced (log-domain column scaling, as in Sinkhorn).

    Counters a classifier that sends most target windows to one class (e.g.
    WALK -> SIT). Fitted on target DEVELOPMENT windows only, never validation.
    """
    s = clf.decision(E_unlab)
    mask = np.isin(clf.classes_, allowed)
    b = np.zeros(len(clf.classes_))
    for _ in range(n_iter):
        p = _softmax(np.where(mask, s + b, -np.inf))
        b[mask] -= np.log(np.maximum(p[:, mask].mean(0) * mask.sum(), 1e-12))
    return b


def make_classifier(name):
    return {"proto": PrototypeClassifier, "ensemble": EnsembleClassifier}.get(name, LogRegClassifier)()


def predict_restricted(clf, E, allowed_per_row):
    """Argmax over allowed classes per row; returns labels and softmax confidence."""
    s = clf.decision(E)
    labels = np.empty(len(E), dtype=object)
    conf = np.empty(len(E))
    for i in range(len(E)):
        allowed = allowed_per_row[i]
        mask = np.isin(clf.classes_, allowed)
        si = np.where(mask, s[i], -np.inf)
        p = np.exp(si - si.max())
        p /= p.sum()
        j = int(np.argmax(p))
        labels[i] = clf.classes_[j]
        conf[i] = p[j]
    return labels, conf


# ------------------------------------------------------------- refinement
def sinkhorn(scores, n_iter=50):
    """Balanced soft assignment: rows sum to 1, columns share mass equally."""
    P = np.exp(scores - scores.max())
    P /= P.sum()
    r = np.full(P.shape[0], 1.0 / P.shape[0])
    c = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(n_iter):
        P *= (r / np.maximum(P.sum(1), 1e-30))[:, None]
        P *= (c / np.maximum(P.sum(0), 1e-30))[None, :]
    return P / P.sum(1, keepdims=True)


def refine_prototypes(clf, E_unlab, allowed, E_sup=None, y_sup=None, iters=10, eps=0.05,
                      support_weight=1.0):
    """Transductive prototype refinement on unlabelled target-development windows.

    Soft k-means initialised from the current prototypes (source-learned shared
    classes + support-learned exclusive classes) with a Sinkhorn balanced-
    assignment step, so that in-domain support prototypes cannot absorb all
    target windows. Each support class keeps an anchor: its support mean is
    added with weight `support_weight` x (that class's assigned mass). Uses no
    validation data and no target labels beyond the K support captures.
    """
    if isinstance(clf, EnsembleClassifier):
        refine_prototypes(clf.parts[0], E_unlab, allowed, E_sup, y_sup, iters, eps, support_weight)
        return clf
    if not isinstance(clf, PrototypeClassifier):
        return clf
    idx = np.where(np.isin(clf.classes_, allowed))[0]
    names = list(clf.classes_[idx])
    En = _l2(E_unlab)
    protos = clf.protos_[idx].copy()
    anchors = {}
    if E_sup is not None:
        Es = _l2(E_sup)
        for j, c in enumerate(names):
            m = np.asarray(y_sup) == c
            if m.any():
                anchors[j] = _l2(Es[m].mean(0))
    for _ in range(iters):
        Q = sinkhorn(En @ protos.T / eps)
        num = Q.T @ En
        mass = Q.sum(0)
        for j, a in anchors.items():
            num[j] += support_weight * mass[j] * a
        protos = _l2(num)
    clf.protos_ = clf.protos_.copy()
    clf.protos_[idx] = protos
    return clf


# ------------------------------------------------------- shared relabelling
MOTION_ORDER = ["SIT", "STAND", "WALK"]  # ascending body-motion prior


def relabel_shared(pred_unl, pred_q, F_unl, F_q, motion_unl, k=9, seed=0):
    """Re-assign shared-activity labels from target-room structure.

    Cross-room transfer of SIT/STAND/WALK fails systematically (WALK <-> SIT),
    while within one room these activities separate well. So the shared label
    is taken from the target room itself: windows the classifier calls shared
    are clustered (k-means on standardised features of unlabelled target-dev
    windows), and each cluster is named by the rank of its median motion score
    under the physical prior WALK > STAND > SIT (k clusters split in tertiles).
    A query window predicted as shared takes its nearest cluster's name.
    k is one cluster count, or several whose relabellings are combined by
    majority vote (ties go to the middle clustering's label).
    No target shared labels and no validation data are used for fitting.
    """
    if np.ndim(k):
        P = np.stack([_relabel_once(pred_unl, pred_q, F_unl, F_q, motion_unl, kk, seed) for kk in k])
        out = P[len(P) // 2].copy()
        for i in range(P.shape[1]):
            labels, counts = np.unique(P[:, i], return_counts=True)
            if counts.max() > 1:
                out[i] = labels[counts.argmax()]
        return out
    return _relabel_once(pred_unl, pred_q, F_unl, F_q, motion_unl, k, seed)


def _relabel_once(pred_unl, pred_q, F_unl, F_q, motion_unl, k, seed):
    pu, pq = np.isin(pred_unl, SHARED), np.isin(pred_q, SHARED)
    out = np.asarray(pred_q).copy()
    if pu.sum() < k or not pq.any():
        return out
    sc = StandardScaler().fit(F_unl[pu])
    km = KMeans(k, n_init=10, random_state=seed).fit(sc.transform(F_unl[pu]))
    med = [np.median(motion_unl[pu][km.labels_ == j]) for j in range(k)]
    names = {j: MOTION_ORDER[rank * 3 // k] for rank, j in enumerate(np.argsort(med))}
    out[pq] = [names[j] for j in km.predict(sc.transform(F_q[pq]))]
    return out


# ------------------------------------------------------------------ episodes
def select_support(meta_dev, target, K, seed):
    """Return capture ids of K support captures per target-exclusive class."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in EXCLUSIVE[target]:
        caps = sorted(meta_dev[(meta_dev.room == target) & (meta_dev.label == c)].capture_id.unique())
        chosen += list(rng.choice(caps, size=min(K, len(caps)), replace=False))
    return chosen


def run_episode(E_dev, meta_dev, E_val, meta_val, target, K, seed, routed_rooms,
                align_stats, classifier="proto", routing="predicted", refine=False, calibrate=False,
                relabel=None):
    """Fit on source labels + K target support captures, predict target validation.

    routing: 'predicted' | 'oracle' | 'none'
    routed_rooms: router output for the validation windows (used if 'predicted').
    Returns dict with predictions and the support capture ids.
    """
    source = [r for r in ROOMS if r != target][0]
    support = select_support(meta_dev, target, K, seed)
    src = (meta_dev.room == source).values
    sup = meta_dev.capture_id.isin(support).values
    train = src | sup

    rooms_dev = meta_dev.room.values
    vm = (meta_val.room == target).values
    if routing == "none":
        A_dev = apply_alignment(E_dev, None, align_stats)
        A_val = apply_alignment(E_val[vm], None, align_stats)
        val_rooms = np.array([None] * vm.sum())
    else:
        val_rooms = meta_val.room.values[vm] if routing == "oracle" else np.asarray(routed_rooms)[vm]
        A_dev = apply_alignment(E_dev, rooms_dev, align_stats)
        A_val = apply_alignment(E_val[vm], val_rooms, align_stats)

    clf = make_classifier(classifier).fit(
        A_dev[train], meta_dev.label.values[train], meta_dev.capture_id.values[train])

    # unlabelled windows recorded during the target-room setup visit
    unl = (meta_dev.room == target).values & ~sup
    allowed = clf.classes_ if routing == "none" else ROOM_ACTIVITIES[target]
    if refine:
        clf = refine_prototypes(clf, A_dev[unl], allowed, A_dev[sup], meta_dev.label.values[sup])
    if calibrate:
        clf = BiasedClassifier(clf, class_bias(clf, A_dev[unl], allowed))

    all_labels = list(clf.classes_)
    allowed_val = [all_labels if r is None else ROOM_ACTIVITIES[r] for r in val_rooms]
    pred, conf = predict_restricted(clf, A_val, allowed_val)
    pred_unl = None
    if relabel is not None:
        # relabel = (F_dev, F_val, motion_dev, k, with_emb): clustering features for all
        # dev/val windows, the dev motion score, cluster count(s), and whether the
        # aligned embedding is appended; only target-room windows are touched
        F_dev, F_val, motion_dev, k, with_emb = relabel
        pred_unl, _ = predict_restricted(clf, A_dev[unl], [allowed] * unl.sum())
        inroom = np.array([r in (None, target) for r in val_rooms])
        F_unl, F_q = F_dev[unl], F_val[vm][inroom]
        if with_emb:
            F_unl, F_q = np.hstack([F_unl, A_dev[unl]]), np.hstack([F_q, A_val[inroom]])
        pred = pred.astype(object)
        pred[inroom] = relabel_shared(pred_unl, pred[inroom], F_unl, F_q, motion_dev[unl], k=k)
    return dict(y_true=meta_val.label.values[vm], y_pred=pred.astype(str), confidence=conf,
                window_id=meta_val.window_id.values[vm], routed_room=val_rooms, support=support,
                pred_unl=pred_unl)
