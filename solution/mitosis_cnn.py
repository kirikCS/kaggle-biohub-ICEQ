#!/usr/bin/env python
"""Cell-state classifier on raw 3D crops over three frames (Linajea / OrganoidTracker style): normal, dividing
parent (the frame before the split) or fresh daughter (the frame after).

    python mitosis_cnn.py crops --data-dir D --out OUT [--neg-per-video 150]   # GT nodes -> crops (npz per video)
    python mitosis_cnn.py train --out OUT --gpu 0                             # leave-one-embryo-out + all-data model

A crop is 3 channels (t-1, t, t+1) x 9 z-slices x 48 x 48 px around the node (±6.5 µm z, ±9.75 µm y/x), each frame
normalised by the video's 1st/99.8th percentiles. Training augments with flips, 90-degree rotations, intensity jitter
and ±2 px shifts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import biohub_clean as bc  # noqa: E402
from div_tally import gt_arrays  # noqa: E402

HZ, HY = 4, 24          # crop half sizes: 9 slices, 48 px (the crop is cut with a 2 px margin for shift jitter)
M = 2


def cut(vol, c, hz=HZ, hy=HY, m=M):
    """(2hz+1, 2(hy+m), 2(hy+m)) crop around voxel centre c, zero padded."""
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in c)
    out = np.zeros((2 * hz + 1, 2 * (hy + m), 2 * (hy + m)), dtype=np.float32)
    z0, y0, x0 = z - hz, y - hy - m, x - hy - m
    zs, ys, xs = max(0, z0), max(0, y0), max(0, x0)
    ze, ye, xe = min(Z, z0 + out.shape[0]), min(Y, y0 + out.shape[1]), min(X, x0 + out.shape[2])
    if ze > zs and ye > ys and xe > xs:
        out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = vol[zs:ze, ys:ye, xs:xe]
    return out


JITTER = False        # also label the parent one frame before the split and the daughters one frame after (metric tolerance)


def video_crops(args):
    name, data_dir, out, neg_per_video, seed = args[:5]
    jitter = args[5] if len(args) > 5 else False
    from evaluate import _load_graph
    rng = np.random.default_rng(seed)
    gt = _load_graph(Path(data_dir) / f"{name}.geff")
    gid, gt_t, gt_um, gt_e = gt_arrays(gt)
    kids, par = {}, {}
    for a, b in gt_e.tolist():
        kids.setdefault(a, []).append(b)
        par[b] = a
    label = {}
    for i in gid.tolist():
        if len(kids.get(i, [])) == 2:
            label[i] = 1                                     # dividing parent
        elif i in par and len(kids.get(par[i], [])) == 2:
            label[i] = 2                                     # fresh daughter
        elif len(kids.get(i, [])) == 1 and i in par:
            label[i] = 0                                     # normal, with a known past and future
    width = int(args[8]) if len(args) > 8 and args[8] else 1
    if jitter:                                               # +-w frames: pre-mitotic parents, young daughters
        for _ in range(max(1, width)):
            for i in [g for g, l in list(label.items()) if l == 1]:
                if i in par and len(kids.get(par[i], [])) == 1:
                    label[par[i]] = 1
            for i in [g for g, l in list(label.items()) if l == 2]:
                for c in kids.get(i, []):
                    if len(kids.get(i, [])) == 1:
                        label[c] = 2
    pos = [i for i, l in label.items() if l > 0]
    neg = [i for i, l in label.items() if l == 0]
    neg = list(rng.choice(neg, size=min(len(neg), neg_per_video), replace=False)) if neg else []
    sel = pos + neg
    if not sel:
        return name, 0
    zarr = Path(data_dir) / f"{name}.zarr"
    T = bc.zarr_meta(zarr)[0][0]
    f0 = bc.read_frame(zarr, 0)
    lo, hi = float(np.percentile(f0, 1.0)), float(np.percentile(f0, 99.8))
    idx = {int(g): k for k, g in enumerate(gid)}
    frames = {}

    def frame(t):
        t = min(max(t, 0), T - 1)
        if t not in frames:
            frames[t] = np.clip((bc.read_frame(zarr, t).astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1.5)
        return frames[t]

    n_frames = args[6] if len(args) > 6 else 3
    hz = args[7] if len(args) > 7 else HZ
    offs = tuple(range(-(n_frames // 2), n_frames // 2 + 1))
    X = np.zeros((len(sel), n_frames, 2 * hz + 1, 2 * (HY + M), 2 * (HY + M)), dtype=np.float16)
    y = np.zeros(len(sel), dtype=np.int8)
    order = sorted(range(len(sel)), key=lambda k: gt_t[idx[int(sel[k])]])
    for k in order:
        g = int(sel[k])
        t = int(gt_t[idx[g]])
        c = gt_um[idx[g]] / np.array(bc.SCALE_UM)
        for ch, dt in enumerate(offs):
            X[k, ch] = cut(frame(t + dt), c, hz=hz)
        y[k] = label[g]
        for old in [f for f in frames if f < t - n_frames // 2]:
            frames.pop(old)
    np.savez_compressed(Path(out) / f"{name}.npz", X=X, y=y)
    return name, len(sel)


def cmd_crops(a):
    out = Path(a.out) / "crops"
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name[:-5] for p in Path(a.data_dir).glob("*.geff"))
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(video_crops, [(n, a.data_dir, str(out), a.neg_per_video, k, a.jitter, a.frames, getattr(a, "hz", HZ),
                                      getattr(a, "jitter_width", 1))
                                     for k, n in enumerate(names)], chunksize=1)
    print(f"{len(res)} videos, {sum(r[1] for r in res)} crops, {time.time() - t0:.0f}s", flush=True)


build_model = bc.mitosis_build_model          # the pipeline module holds the architecture


def save_state(model, path, hz):
    """State dict plus the crop half depth, so the pipeline cuts crops of the depth the model was trained on."""
    import torch
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    sd["_hz"] = torch.tensor(int(hz))
    torch.save(sd, path)


AUG = "base"          # "strong": adds gamma, noise, in-plane rescale and time reversal (embryo-robustness)
NORM = "video"        # "crop": per-crop standardisation (see biohub_clean.mitosis_norm_crops)


def augment(x, rng, labels=None):
    """x: (B, 3, Z, H+2M, W+2M) float tensor -> random flips / rot90 / intensity / shift, cropped to (Z, H, W)."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    out = []
    for b in range(B):
        v = x[b]
        if rng.random() < 0.5:
            v = torch.flip(v, dims=[3])
        if rng.random() < 0.5:
            v = torch.flip(v, dims=[2])
        if rng.random() < 0.5:
            v = torch.flip(v, dims=[1])
        v = torch.rot90(v, int(rng.integers(0, 4)), dims=[2, 3])
        if AUG == "strong":
            if labels is not None and int(labels[b]) == 0 and rng.random() < 0.5:   # time reversal only for normal cells
                v = torch.flip(v, dims=[0])
            if rng.random() < 0.7:                        # in-plane size jitter (nuclei of another embryo / stage)
                f = float(rng.uniform(0.85, 1.18))
                H = v.shape[-1]
                v = F.interpolate(v[None], scale_factor=(1.0, f, f), mode="trilinear", align_corners=False)[0]
                if v.shape[-1] < H:
                    pad = H - v.shape[-1]
                    v = F.pad(v, (0, pad, 0, pad))
                else:
                    o = (v.shape[-1] - H) // 2
                    v = v[:, :, o:o + H, o:o + H]
            if rng.random() < 0.5:                        # gamma
                v = v.clamp_min(0) ** float(rng.uniform(0.7, 1.4))
        dy, dx = rng.integers(0, 2 * M + 1, size=2)
        v = v[:, :, dy:dy + 2 * HY, dx:dx + 2 * HY]
        v = v * float(rng.uniform(0.8, 1.25)) + float(rng.uniform(-0.05, 0.05))
        if AUG == "strong" and rng.random() < 0.5:
            v = v + torch.randn_like(v) * float(rng.uniform(0.0, 0.06))
        out.append(v)
    return torch.stack(out)


def center(x):
    return x[:, :, :, M:M + 2 * HY, M:M + 2 * HY]


def train_one(Xtr, ytr, Xte, yte, device, epochs, seed, log):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(int(Xtr.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * max(1, len(ytr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    # balanced sampling: half of every batch from the two rare classes
    cls = [np.nonzero(ytr == c)[0] for c in range(3)]
    w = torch.tensor([1.0, 1.0, 1.0], device=device)
    step = 0
    model.train()
    while step < steps:
        nb = 128
        idx = np.concatenate([rng.choice(cls[0], nb // 2), rng.choice(cls[1], nb // 4) if len(cls[1]) else [],
                              rng.choice(cls[2], nb // 4) if len(cls[2]) else []]).astype(int)
        xb = torch.as_tensor(Xtr[idx].astype(np.float32), device=device)
        xb = augment(xb, rng, ytr[idx])
        if NORM == "crop":
            xb = bc.mitosis_norm_crops(xb)
        yb = torch.as_tensor(ytr[idx].astype(np.int64), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 500 == 0:
            log(f"   step {step}/{steps} loss {loss.item():.3f}")
    model.eval()
    probs = predict(model, Xte, device) if Xte is not None else None
    return model, probs


def predict(model, X, device, tta=True):
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 256):
            xb = center(torch.as_tensor(X[s:s + 256].astype(np.float32), device=device))
            if NORM == "crop":
                xb = bc.mitosis_norm_crops(xb)
            views = [xb, torch.flip(xb, dims=[4]), torch.flip(xb, dims=[3]), torch.rot90(xb, 1, dims=[3, 4])] if tta else [xb]
            p = sum(torch.softmax(model(v).float(), dim=1) for v in views) / len(views)
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def cmd_train(a):
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    global AUG, NORM
    AUG = getattr(a, "aug", "base")
    NORM = getattr(a, "norm", "video")
    device = f"cuda:{a.gpu}"
    data = {}
    for p in sorted(Path(a.out, "crops").glob("*.npz")):
        d = np.load(p)
        data[p.stem] = (d["X"], d["y"])
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    extra = [(np.load(p)["X"], np.load(p)["y"]) for p in sorted(Path(a.extra_crops, "crops").glob("*.npz"))] if getattr(a, "extra_crops", "") else []
    if extra:
        log(f"extra crops: {sum(len(e[1]) for e in extra)} from {a.extra_crops}")

    def stack(emb, with_extra=False):
        ks = [k for k in data if k.startswith(emb)] if emb else list(data)
        ex = extra if with_extra else []
        return (np.concatenate([data[k][0] for k in ks] + [e[0] for e in ex]),
                np.concatenate([data[k][1] for k in ks] + [e[1] for e in ex]))
    for tr_e, te_e in (("44b6", "6bba"), ("6bba", "44b6")):
        Xtr, ytr = stack(tr_e, True)
        Xte, yte = stack(te_e)
        log(f"{tr_e}->{te_e}: train {np.bincount(ytr, minlength=3).tolist()} test {np.bincount(yte, minlength=3).tolist()}")
        model, pr = train_one(Xtr, ytr, Xte, yte, device, a.epochs, 0, log)
        save_state(model, Path(a.out) / f"mitosis_{tr_e}.pt", Xtr.shape[2] // 2)
        for c, nm in ((1, "parent"), (2, "daughter")):
            m = (yte == 0) | (yte == c)
            if (yte[m] == c).sum():
                log(f"   {nm} vs normal: AUC {roc_auc_score(yte[m] == c, pr[m, c]):.3f} "
                    f"AP {average_precision_score(yte[m] == c, pr[m, c]):.3f} (n={int((yte[m] == c).sum())})")
    Xall, yall = stack(None, True)
    model, _ = train_one(Xall, yall, None, None, device, a.epochs, 100 * getattr(a, "seed", 0), log)
    save_state(model, Path(a.out) / f"mitosis_all{getattr(a, 'tag', '')}.pt", Xall.shape[2] // 2)
    (Path(a.out) / "report.txt").write_text("\n".join(lines))


def load_model(path, device):
    """One checkpoint or a comma-separated ensemble (see biohub_clean.mitosis_load)."""
    return bc.mitosis_load(path, device)


def score_nodes(model, zarr_path, txyz, device, batch=384, views=4):
    return bc.mitosis_score_nodes(model, zarr_path, txyz, device, batch, views, norm=NORM)


def cmd_score(a):
    """Scores of every cached detection per video, with the model trained on the other embryo (or --model)."""
    device = f"cuda:{a.gpu}"
    out = Path(a.out) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    names = sorted(p.stem for p in Path(a.k0, "cache").glob("*.npz"))
    if a.shard is not None:
        k, n = (int(v) for v in a.shard.split("/"))
        names = names[k::n]
    t0 = time.time()
    other = {"44b6": "6bba", "6bba": "44b6"}
    for i, name in enumerate(names):
        if (out / f"{name}.npz").exists():
            continue
        path = a.model or str(Path(a.out) / f"mitosis_{other.get(name[:4], 'all')}.pt")
        if path not in models:
            models[path] = load_model(path, device)
        cache = bc.RichCache.load(Path(a.k0) / "cache" / f"{name}.npz")
        mit = score_nodes(models[path], Path(a.data_dir) / f"{name}.zarr", cache.txyz, device)
        np.savez_compressed(out / f"{name}.npz", mit=mit.astype(np.float16))
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(names)} videos, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(names)} videos, {time.time() - t0:.0f}s", flush=True)


def cmd_cv(a):
    """Video-level K-fold: every video is scored by a model that never saw it (honest scores for the public lab)."""
    import torch
    device = f"cuda:{a.gpu}"
    data = {p.stem: (np.load(p)["X"], np.load(p)["y"]) for p in sorted(Path(a.out, "crops").glob("*.npz"))}
    names = sorted(data)
    rng = np.random.default_rng(0)
    folds = {}
    global AUG, NORM
    AUG = a.aug
    NORM = getattr(a, "norm", "video")
    tag = ("loeo" if a.loeo else "cv") + a.tag
    if a.loeo:                                       # embryo-disjoint: the strict judge for a test of other embryos
        a.folds = 2
        folds = {n: int(n.startswith("6bba")) for n in names}
    else:
        for emb in ("44b6", "6bba"):                 # stratified by embryo
            vids = [n for n in names if n.startswith(emb)]
            rng.shuffle(vids)
            for i, v in enumerate(vids):
                folds[v] = i % a.folds
    out = Path(a.out) / f"scores_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{tag}_folds.json").write_text(json.dumps(folds, indent=1))
    lines = []
    extra = [(np.load(p)["X"], np.load(p)["y"]) for p in sorted(Path(a.extra_crops, "crops").glob("*.npz"))] if getattr(a, "extra_crops", "") else []
    if extra:
        print(f"extra crops: {sum(len(e[1]) for e in extra)} from {a.extra_crops}", flush=True)
    for k in range(a.folds):
        tr = [n for n in names if folds[n] != k]
        Xtr = np.concatenate([data[n][0] for n in tr] + [e[0] for e in extra]); ytr = np.concatenate([data[n][1] for n in tr] + [e[1] for e in extra])
        model, _ = train_one(Xtr, ytr, None, None, device, a.epochs, k + 100 * a.seed, lambda s: None)
        model.mit_hz = Xtr.shape[2] // 2
        save_state(model, Path(a.out) / f"mitosis_{tag}{k}.pt", model.mit_hz)
        held = [n for n in sorted(p.stem for p in Path(a.k0, "cache").glob("*.npz")) if folds.get(n) == k]
        for name in held:
            cache = bc.RichCache.load(Path(a.k0) / "cache" / f"{name}.npz")
            ns = cache.n_strong
            mit = np.full((len(cache.txyz), 3), np.nan, dtype=np.float32)
            mit[:ns] = score_nodes(model, Path(a.data_dir) / f"{name}.zarr", cache.txyz[:ns], device)
            np.savez_compressed(out / f"{name}.npz", mit=mit.astype(np.float16))
        lines.append(f"fold {k}: trained on {len(tr)} videos, scored {len(held)}")
        print(lines[-1], flush=True)
    (Path(a.out) / f"{tag}_report.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("crops")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--neg-per-video", type=int, default=150)
    c.add_argument("--jitter", action="store_true", help="label t-1 parents and t+2 daughters as positives too")
    c.add_argument("--jitter-width", type=int, default=1, help="with --jitter: how many frames back (parents) / forward (daughters)")
    c.add_argument("--frames", type=int, default=3, choices=(3, 5), help="frames stacked as channels around t")
    c.add_argument("--hz", type=int, default=HZ, help="crop half depth in z-slices (4 = 9 slices, 7 = 15 slices)")
    c.add_argument("--workers", type=int, default=os.cpu_count())
    t = sp.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--aug", default="base", choices=("base", "strong"))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--norm", default="video", choices=("video", "crop"))
    t.add_argument("--extra-crops", default="")
    t.add_argument("--tag", default="", help="suffix of the saved model name (mitosis_all<tag>.pt)")
    sc = sp.add_parser("score")
    sc.add_argument("--k0", required=True)
    sc.add_argument("--data-dir", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--gpu", type=int, default=0)
    sc.add_argument("--model", default=None)
    sc.add_argument("--shard", default=None, help="k/n: every n-th video starting at k")
    cv = sp.add_parser("cv")
    cv.add_argument("--out", required=True)
    cv.add_argument("--k0", required=True, help="caches whose strong detections get honest scores")
    cv.add_argument("--data-dir", required=True)
    cv.add_argument("--gpu", type=int, default=0)
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=30)
    cv.add_argument("--loeo", action="store_true", help="2 folds by embryo -> scores_loeo (strict judge)")
    cv.add_argument("--aug", default="base", choices=("base", "strong"))
    cv.add_argument("--tag", default="", help="suffix for the fold files / score dir (e.g. _strong)")
    cv.add_argument("--seed", type=int, default=0, help="training seed offset (ensembles: 1, 2, ...)")
    cv.add_argument("--norm", default="video", choices=("video", "crop"))
    cv.add_argument("--extra-crops", default="", help="dir with crops/*.npz added to every training fold (e.g. Zebrahub)")
    a = ap.parse_args()
    {"crops": cmd_crops, "train": cmd_train, "score": cmd_score, "cv": cmd_cv}[a.cmd](a)


if __name__ == "__main__":
    main()
