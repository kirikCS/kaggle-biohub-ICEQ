#!/usr/bin/env python
"""Where do ground-truth divisions get lost?  One row per GT division of every OOF video.

For each GT division (parent P at t, daughters D1/D2 at t+1) the repaired graph of a lab variant is scored with the
official division metric (TP or not) and, when missed, the first blocking step is recorded:
  parent_unmatched / daughter_missing (strong peak dropped | weak peak only | no peak) / parent_links_elsewhere /
  parent_track_ends / second_daughter_linked_elsewhere / second_daughter_orphan (+ which safe-division gate fails) /
  fork_present_not_tp.
It also stores the transformer's base link probabilities P->D1 and P->D2 between the cached detections matched to the
GT nodes, to see whether the network itself sees the second daughter.

    python div_tally.py --folds /workspace/biohub/k0 --data-dir /workspace/biohub/data/train --out OUT [--overrides '{}']
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pp_lab  # noqa: E402
import biohub_clean as bc  # noqa: E402

MATCH_UM = 7.0
SCALE = np.array(bc.SCALE_UM)


def gt_arrays(gt):
    na = gt.node_attrs().select(["node_id", "t", "z", "y", "x"]).to_numpy()
    ids = na[:, 0].astype(np.int64)
    t = na[:, 1].astype(np.int64)
    um = na[:, 2:5].astype(np.float64) * SCALE[None, :]
    ea = gt.edge_attrs().select(["source_id", "target_id"]).to_numpy().astype(np.int64)
    return ids, t, um, ea


def match_frames(a_t, a_um, b_t, b_um, max_d=MATCH_UM):
    """Optimal per-frame assignment a -> b within max_d µm (same rule as the metric); returns {a_index: b_index}."""
    out = {}
    for t in np.unique(a_t):
        ia = np.nonzero(a_t == t)[0]
        ib = np.nonzero(b_t == t)[0]
        if not len(ia) or not len(ib):
            continue
        d = np.linalg.norm(a_um[ia][:, None, :] - b_um[ib][None, :, :], axis=2)
        cost = np.where(d <= max_d, d, 1e6)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] < 1e6:
                out[int(ia[i])] = int(ib[j])
    return out


def base_prob(cache, s, g):
    """Transformer probability of the link s -> g among g's strong parents (base regime); nan if not a candidate."""
    if s is None or g is None or cache.base_src is None:
        return float("nan")
    m = np.nonzero((cache.base_src == s) & (cache.base_tgt == g))[0]
    if not len(m):
        return float("nan")
    return float(np.exp(cache.base_logit[m[0]] - cache.base_lse[g]))


def nearest_cache(cache, t, p_um, radius, kind):
    ids, d = cache.near(int(t), p_um, radius, kind)
    return (int(ids[0]), float(d[0])) if len(ids) else (None, float("nan"))


def analyse_video(name, overrides, paths):
    from evaluate import _load_graph
    from tracking_cellmot.division_metrics import score_divisions, extract_divisions
    cfg = pp_lab.make_config(overrides)
    nodes, edges, _ = pp_lab.run_variant(name, cfg, paths)
    cache = pp_lab.Cache.load(Path(paths["cache_dir"]) / f"{name}.npz")
    gt_path = Path(paths["data_dir"]) / f"{name}.geff"
    gt = _load_graph(gt_path)
    pred = pp_lab.pred_graph(name, nodes, edges)
    res = score_divisions(pred, _load_graph(gt_path), tuple(SCALE), MATCH_UM)
    gdiv = extract_divisions(_load_graph(gt_path))

    gid, gt_t, gt_um, gt_e = gt_arrays(gt)
    gpos = {int(i): k for k, i in enumerate(gid)}
    g_children, g_parent = {}, {}
    for s, g in gt_e:
        g_children.setdefault(int(s), []).append(int(g))
        g_parent[int(g)] = int(s)

    pid = np.array(list(nodes.keys()), dtype=np.int64)
    p_t = np.array([nodes[i]["t"] for i in pid], dtype=np.int64)
    p_um = np.array([bc.point_um(nodes[i]) for i in pid], dtype=np.float64)
    m = match_frames(gt_t, gt_um, p_t, p_um)
    gt2pred = {int(gid[a]): int(pid[b]) for a, b in m.items()}
    p_children, p_parent = {}, {}
    for e in edges:
        p_children.setdefault(e["source_id"], []).append(e["target_id"])
        p_parent[e["target_id"]] = e["source_id"]
    orphans_by_t = {}
    for i in pid:
        if int(i) not in p_parent:
            orphans_by_t.setdefault(int(nodes[int(i)]["t"]), []).append(int(i))

    rows = []
    for div_node, score in res.scores.items():
        P = int(div_node)
        kids = sorted(g_children.get(P, []), key=lambda k: gt_um[gpos[k]][0])
        if len(kids) < 2:
            continue
        D1, D2 = kids[0], kids[1]
        t = int(gt_t[gpos[P]])
        Pu, D1u, D2u = gt_um[gpos[P]], gt_um[gpos[D1]], gt_um[gpos[D2]]
        v1, v2 = D1u - Pu, D2u - Pu
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / max(1e-9, np.linalg.norm(v1) * np.linalg.norm(v2))))))
        row = {"video": name, "embryo": name.split("_")[0], "t": t, "T": int(cache.T), "tp": int(score),
               "gt_d_p_d1": float(np.linalg.norm(v1)), "gt_d_p_d2": float(np.linalg.norm(v2)),
               "gt_d_sisters": float(np.linalg.norm(D1u - D2u)), "gt_angle": ang,
               "gt_parent_has_parent": int(P in g_parent)}
        # cached detections at the GT positions (strong peaks) and the network's own link probabilities
        cP, dP = nearest_cache(cache, t, Pu, MATCH_UM, "strong")
        c1, d1 = nearest_cache(cache, t + 1, D1u, MATCH_UM, "strong")
        c2, d2 = nearest_cache(cache, t + 1, D2u, MATCH_UM, "strong")
        row.update({"cache_P": cP is not None, "cache_D1": c1 is not None, "cache_D2": c2 is not None,
                    "p_P_D1": base_prob(cache, cP, c1), "p_P_D2": base_prob(cache, cP, c2)})
        if c1 is not None and c2 is not None and c1 == c2:
            row["same_cache_daughter"] = 1
        # best parent of each daughter according to the network
        for tag, c in (("D1", c1), ("D2", c2)):
            if c is not None and cache.base_tgt is not None:
                mm = np.nonzero(cache.base_tgt == c)[0]
                if len(mm):
                    probs = np.exp(cache.base_logit[mm] - cache.base_lse[c])
                    k = int(np.argmax(probs))
                    row[f"best_parent_is_P_{tag}"] = int(cache.base_src[mm[k]] == cP)
                    row[f"best_parent_prob_{tag}"] = float(probs[k])
        Pp, D1p, D2p = gt2pred.get(P), gt2pred.get(D1), gt2pred.get(D2)
        if score:
            row["reason"] = "TP"
        elif Pp is None:
            row["reason"] = "parent_unmatched"
        elif D1p is None or D2p is None:
            if D1p is None and D2p is None:
                row["reason"] = "both_daughters_missing"
            else:
                miss_u = D1u if D1p is None else D2u
                cs, _ = nearest_cache(cache, t + 1, miss_u, MATCH_UM, "strong")
                cw, _ = nearest_cache(cache, t + 1, miss_u, MATCH_UM, "weak")
                row["reason"] = ("daughter_missing_strong_dropped" if cs is not None else
                                 "daughter_missing_weak_only" if cw is not None else "daughter_missing_no_peak")
        else:
            ch = set(p_children.get(Pp, []))
            if D1p in ch and D2p in ch:
                row["reason"] = "fork_present_not_tp"
            elif D1p in ch or D2p in ch:
                other = D2p if D1p in ch else D1p
                kept = D1p if D1p in ch else D2p
                if other in p_parent:
                    x = p_parent[other]
                    row["reason"] = "second_daughter_linked_elsewhere"
                    row["thief_dist_to_P"] = float(np.linalg.norm(bc.point_um(nodes[x]) - bc.point_um(nodes[Pp])))
                    row["thief_is_gt"] = int(any(v == x for v in gt2pred.values()))
                else:
                    row["reason"] = "second_daughter_orphan"
                    src_um = bc.point_um(nodes[Pp])
                    q_um, k_um = bc.point_um(nodes[other]), bc.point_um(nodes[kept])
                    d_child, d_parent = float(np.linalg.norm(k_um - src_um)), float(np.linalg.norm(q_um - src_um))
                    d_sister = float(np.linalg.norm(k_um - q_um))
                    gates = []
                    if len(p_children.get(Pp, [])) != 1:
                        gates.append("parent_not_single_child")
                    if d_child > cfg.safe_div_existing_child_um:
                        gates.append("existing_child_far")
                    if d_parent > cfg.safe_div_parent_um:
                        gates.append("parent_dist")
                    if d_sister > cfg.safe_div_sister_um:
                        gates.append("sister_dist")
                    orph = [o for o in orphans_by_t.get(t + 1, [])]
                    if orph:
                        od = [np.linalg.norm(bc.point_um(nodes[o]) - k_um) for o in orph]
                        if orph[int(np.argmin(od))] != other:
                            gates.append("not_nearest_orphan")
                    s1, s2 = p_children.get(kept, []), p_children.get(other, [])
                    if len(s1) != 1 or len(s2) != 1:
                        gates.append("diverge_no_single_successor")
                    else:
                        g1, g2 = nodes[s1[0]], nodes[s2[0]]
                        if bc.dist_um(g1, g2) - d_sister < (cfg.safe_div_diverge_um or 0):
                            gates.append("diverge_small")
                    if cfg.safe_div_symmetry_tau > 0 and \
                            abs(d_child - d_parent) / max((d_child + d_parent) / 2.0, 1e-6) > cfg.safe_div_symmetry_tau:
                        gates.append("asymmetric")
                    row.update({"gates": gates, "d_child": d_child, "d_parent": d_parent, "d_sister": d_sister,
                                "orphan_track_len": track_len(other, p_children)})
            else:
                row["reason"] = "parent_links_elsewhere" if p_children.get(Pp) else "parent_track_ends"
        rows.append(row)
    n_fp = len(res.fp_forks)
    return {"video": name, "rows": rows, "fp_forks": n_fp, "tp_forks": len(res.tp_forks),
            "n_pred_forks": sum(1 for i in p_children if len(p_children[i]) >= 2)}


def track_len(start, children, cap=60):
    n, cur = 1, start
    while cur in children and len(children[cur]) == 1 and n < cap:
        cur = children[cur][0]
        n += 1
    return n


def _task(args):
    name, overrides, paths = args
    try:
        return analyse_video(name, overrides, paths)
    except Exception as exc:
        return {"video": name, "error": str(exc), "trace": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="dir with fold0/ and fold1/ (each with cache/ and raw/)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    overrides = json.loads(a.overrides)
    tasks = []
    subdirs = ("fold0", "fold1") if (Path(a.folds) / "fold0").exists() else (".",)
    for f in subdirs:
        paths = {"raw_dir": f"{a.folds}/{f}/raw", "cache_dir": f"{a.folds}/{f}/cache", "data_dir": a.data_dir}
        names = pp_lab.list_videos(paths, None)
        tasks += [(n, overrides, paths) for n in names]
    if a.limit:
        tasks = tasks[:a.limit]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = list(pool.imap_unordered(_task, tasks, chunksize=1))
    errs = [r for r in res if "error" in r]
    for r in errs[:5]:
        print("ERROR", r["video"], r["error"], r["trace"][-600:])
    rows = [x for r in res if "rows" in r for x in r["rows"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "div_rows.json").write_text(json.dumps(rows, indent=0, default=float))
    lines = [f"videos {len(res)} ({len(errs)} failed), GT divisions {len(rows)}, {time.time() - t0:.0f}s",
             f"pred forks {sum(r.get('n_pred_forks', 0) for r in res)}, TP forks {sum(r.get('tp_forks', 0) for r in res)}, "
             f"FP forks {sum(r.get('fp_forks', 0) for r in res)}"]
    for emb in ("ALL", "44b6", "6bba"):
        sub = [r for r in rows if emb == "ALL" or r["embryo"] == emb]
        c = Counter(r["reason"] for r in sub)
        lines.append(f"\n[{emb}] {len(sub)} divisions: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    gates = Counter(g for r in rows if r["reason"] == "second_daughter_orphan" for g in r.get("gates", []))
    lines.append("\norphan second daughter, failing gates: " + ", ".join(f"{k} {v}" for k, v in gates.most_common()))
    lines.append("orphan with no failing gate (lost to caps/order/DeepCenter): "
                 f"{sum(1 for r in rows if r['reason'] == 'second_daughter_orphan' and not r.get('gates'))}")
    for key in ("gt_d_p_d1", "gt_d_p_d2", "gt_d_sisters", "gt_angle", "p_P_D1", "p_P_D2"):
        v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
        if len(v):
            lines.append(f"{key}: n={len(v)} median {np.median(v):.2f} p10 {np.percentile(v, 10):.2f} p90 {np.percentile(v, 90):.2f}")
    both = [r for r in rows if r.get("cache_P") and r.get("cache_D1") and r.get("cache_D2")]
    lines.append(f"\nGT divisions with all three cached strong peaks: {len(both)}")
    for key in ("best_parent_is_P_D1", "best_parent_is_P_D2"):
        v = [r[key] for r in both if key in r]
        lines.append(f"{key}: {np.mean(v) if v else float('nan'):.3f} of {len(v)}")
    txt = "\n".join(lines)
    (out / "div_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
