"""Shared stage helpers: normalised data, encoders, cached representations."""
import json

import numpy as np
import torch

from .config import resolve
from .dataset import load_split
from .features import handcrafted, motion_features
from .models import CSIEncoder
from .normalization import Normalizer
from .ssl import embed, train_ssl


def artifacts_dir(cfg):
    d = resolve(cfg, "artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_normalizer(cfg, Xd, md):
    nc = cfg["normalization"]
    norm = Normalizer(nc["per_packet_gain"], nc["log"], nc["scope"], nc["null_bin_rel_level"])
    norm.fit(Xd, md.room.values)  # development windows only
    norm.save(artifacts_dir(cfg) / "normalizer.json")
    return norm


def load_normalized(cfg, refit=False):
    """Globally normalised windows (no room labels needed at transform time)."""
    Xd, md = load_split(cfg, "development")
    Xv, mv = load_split(cfg, "validation")
    path = artifacts_dir(cfg) / "normalizer.json"
    norm = fit_normalizer(cfg, Xd, md) if refit or not path.exists() else Normalizer.load(path)
    Zd = norm.transform(Xd, scope="global")
    Zv = norm.transform(Xv, scope="global")
    return Zd, md, Zv, mv, norm


def get_encoder(cfg, Zd=None, md=None, retrain=False, log=print):
    sc = cfg["ssl"]
    path = artifacts_dir(cfg) / "encoder_ssl.pt"
    enc = CSIEncoder(256, sc["width"], sc["emb_dim"])
    if path.exists() and not retrain:
        enc.load_state_dict(torch.load(path))
        enc.eval()
        return enc
    enc, hist = train_ssl(Zd, md.capture_id.values, cfg, log=log)
    torch.save(enc.state_dict(), path)
    (artifacts_dir(cfg) / "ssl_history.json").write_text(json.dumps(hist))
    return enc


def random_encoder(cfg, seed=0):
    torch.manual_seed(seed)
    sc = cfg["ssl"]
    enc = CSIEncoder(256, sc["width"], sc["emb_dim"])
    enc.eval()
    return enc


def _dynamic(Z):
    """Remove each window's static per-subcarrier profile (the location cue)."""
    return Z - Z.mean(1, keepdims=True)


def representations(cfg, Zd, Zv, md, kinds=("handcrafted", "random", "ssl"), refresh=False, log=print):
    """Return {kind: (E_dev, E_val)}, cached in artifacts/repr_<kind>.npz."""
    out = {}
    for kind in kinds:
        path = artifacts_dir(cfg) / f"repr_{kind}.npz"
        if path.exists() and not refresh:
            d = np.load(path)
            out[kind] = (d["dev"], d["val"])
            continue
        if kind == "handcrafted":
            Ed, Ev = handcrafted(Zd), handcrafted(Zv)
        elif kind == "motion":
            Ed, Ev = motion_features(Zd), motion_features(Zv)
        elif kind == "ssl_dyn":
            enc = get_encoder(cfg, Zd, md, log=log)
            Ed, Ev = embed(enc, _dynamic(Zd)), embed(enc, _dynamic(Zv))
        elif kind == "random":
            enc = random_encoder(cfg)
            Ed, Ev = embed(enc, Zd), embed(enc, Zv)
        else:
            enc = get_encoder(cfg, Zd, md, log=log)
            Ed, Ev = embed(enc, Zd), embed(enc, Zv)
        np.savez(path, dev=Ed, val=Ev)
        out[kind] = (Ed, Ev)
    return out
