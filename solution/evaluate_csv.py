#!/usr/bin/env python
"""Score submission-format CSVs against train ground truth with the organizers' official metric code.

Uses tracking_cellmot (royerlab/kaggle-cell-tracking-competition, commit 075fc5f "metrics-fix"):
per-video evaluate() + per_sample_metrics() with estimated_number_of_nodes, then summarise().
Reports the overall score and the same summary per embryo (dataset name prefix).

    python evaluate_csv.py --gt-dir /workspace/biohub/data/train --csv a.csv [b.csv ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("BIOHUB_METRIC_REPO", "/workspace/biohub/kaggle-cell-tracking-competition"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from csv_to_geffs import build_graph_from_rows  # noqa: E402
from evaluate import _load_graph, _read_estimated_n_total, _read_scale  # noqa: E402
from tracking_cellmot.metrics import evaluate as compute_metric, node_recall, per_sample_metrics, summarise  # noqa: E402


def score_csv(csv_path: Path, gt_dir: Path, only: set[str] | None = None) -> list[dict]:
    df = pl.read_csv(csv_path, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x",
                                        "source_id", "target_id"])
    rows = []
    for (name,), group in sorted(df.group_by("dataset"), key=lambda kv: kv[0][0]):
        if only and name not in only:
            continue
        gt_path = gt_dir / f"{name}.geff"
        if not gt_path.exists():
            continue
        pred = build_graph_from_rows(group.filter(pl.col("row_type") == "node"),
                                     group.filter(pl.col("row_type") == "edge"))
        gt = _load_graph(gt_path)
        er = compute_metric(pred, gt, scale=_read_scale(gt_dir, name), max_distance=7.0)
        rec = node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
        row = per_sample_metrics(er, _read_estimated_n_total(gt_path), rec)
        row["dataset"] = name
        rows.append(row)
    return rows


def fmt(s: dict) -> str:
    return (f"score={s['score']:.4f}  adjJ={s['adj_edge_jaccard']:.4f}  edgeJ={s['edge_jaccard']:.4f}  "
            f"divJ={s['division_jaccard']:.4f} (TP/FP/FN {s['division_tp']}/{s['division_fp']}/{s['division_fn']})  "
            f"recall={s['node_recall']:.4f}  n={s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    ap.add_argument("--videos", default=None, help="comma-separated subset")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--per-video", action="store_true")
    args = ap.parse_args()
    only = set(args.videos.split(",")) if args.videos else None
    report = {}
    for csv_path in args.csv:
        rows = score_csv(csv_path, args.gt_dir, only)
        overall = summarise(rows)
        by_emb = {}
        for emb in sorted({r["dataset"].split("_")[0] for r in rows}):
            by_emb[emb] = summarise([r for r in rows if r["dataset"].startswith(emb + "_")])
        print(f"\n== {csv_path}")
        print(f"   ALL   {fmt(overall)}")
        for emb, s in by_emb.items():
            print(f"   {emb}  {fmt(s)}")
        if args.per_video:
            for r in rows:
                print(f"     {r['dataset']}: adjJ={r['adj_edge_jaccard']:.4f} edge TP/FP/FN "
                      f"{r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']} div {r['division_tp']}/{r['division_fp']}/"
                      f"{r['division_fn']} n_pred={r['num_pred_nodes']} ratio={r['total_node_ratio']:+.3f}")
        report[str(csv_path)] = {"overall": overall, "by_embryo": by_emb, "rows": rows}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
