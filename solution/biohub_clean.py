#!/usr/bin/env python
"""Clean inference pipeline for Kaggle "Biohub - Cell Tracking During Development".

Lineage
    Model core: pilkwang's public support pack (TemporalUNet3D detector + node-transformer edge scorer,
    tracksdata ILP), second seed (seed314159) and the DeepCenter center prior.
    Association and graph repair: the public 0.94x "harmonic fusion" notebooks (reference: #5
    flexonafft/biohub-harmonic-fusion).

What this file keeps
    8-view D4 TTA for detection and edge features, dual-seed detection fusion with a retention guard,
    harmonic bidirectional edge fusion, low-margin secondary edge consensus, ILP, motion relink,
    gap-1 / gap-2 closing, safe divisions with DeepCenter confirmation, short-track filter, line-fit smoothing.

What this file removes or fixes (see CHANGES.md for the full list)
    - TTA: the public "anti-transpose" view equals the x-flip, so only 7 distinct views are averaged and the
      x-flip counts twice. The models were trained with flips only, and on training videos the true D4 set scores
      lower (0.9356 vs 0.9511) while flip-only sets score higher (0.9568-0.9600). Default keeps the public list
      (``d4_legacy``) until held-out validation decides; ``d4``, ``d4_7``, ``flips8``, ``flips4`` are options.
    - No in-notebook validator / PP sweep (it ran on in-sample train videos with the pre-patch division metric).
    - No recursive globs over /kaggle/input, no env-var configuration, no audit receipts, no string patching
      of the support-pack script, no dead switches (single-child repair, division-geometry filter, gap_max=2).
    - Repair micro-features with no measurable effect removed from the defaults: density-adaptive gap gate,
      adaptive short-track rescue (kept as options for exact reproduction of the public notebook).

Usage
    python biohub_clean.py run --config config.json --data-dir <dir with *.zarr> --out-csv part0.csv \
        [--shard 0 --nshards 2] [--raw-dir raw/] [--videos a,b,c]
    python biohub_clean.py merge --parts part0.csv part1.csv --data-dir <dir> --out submission.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCALE_UM = (1.625, 0.40625, 0.40625)
ROW_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


# =============================================================================================
# Configuration
# =============================================================================================

@dataclass
class Config:
    # ---- inputs (Kaggle layout by default) ----
    support_pack: str = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
    secondary_pack: str = "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1"
    deepcenter_ckpt: str = ("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/"
                            "weights/full_frame_center/best.pt")
    use_secondary: bool = True

    # ---- detection ----
    det_threshold: float = 0.965          # sigmoid prob of a local max; public lineage value
    pool_kernel_um: float = 3.0           # local-max suppression radius (support-pack default)
    det_calib_ref: float = 0.0            # lab: per-video detector calibration; >0 = in-sample reference share of peaks (>= cache_low_thr)
                                          # that are strong; an under-confident video lowers its threshold until its share matches
    det_calib_mode: str = "lower"         # "lower": only admit weak peaks; "two": also drop the weakest strong peaks of over-confident videos
    det_calib_min: float = 0.85           # lowest per-video threshold the rule may pick
    det_calib_max: float = 0.995          # highest (two-sided mode)
    det_phases: int = 1                   # >1: also detect on phase-shifted strided grids (2 px in y/x) and add the peaks the
                                          # phase-0 grid missed (single view, primary detector); 2 = one extra phase, 4 = three
    det_phase_threshold: float = 0.0      # detection threshold on the shifted grids (0 = det_threshold)
    det_phase_merge_um: float = 0.0       # a shifted-grid peak within this distance of a phase-0 peak is a duplicate (0 = pool_kernel_um)
    # ---- train duplicates: a test video that is a copy of a train recording gets the train annotation instead of inference ----
    dup_sigs_path: str = ""               # train frame signatures (train_sigs.py output); empty = off
    dup_train_dir: str = ""               # dir with <name>.zarr + <name>.geff of the train videos; empty = <data_dir>/../train
    dup_ncc_min: float = 0.99             # every test frame must match its train frame at this signature NCC (cross-video max 0.965)
    dup_full_ncc_min: float = 0.995       # and the full-resolution NCC of the checked frames must reach this
    dup_min_cover: float = 1.0            # share of the test frames that must have a train counterpart at the common offset
    tta: str = "d4_legacy"                # model TTA views: d4_legacy | d4 | d4_7 | flips8 | flips4 | none
    secondary_det_weight: float = 0.80   # blend of the aligned second-seed detection logits
    det_retention_guard: float = 0.90    # fall back to primary if blending drops >10% of peaks

    # ---- association ----
    edge_feature_tta: bool = True         # average UNet features over the TTA views for the edge model
    secondary_feature_tta_weight: float = 0.75  # public value; 1.0 measured -0.002 adjJ in-sample
    bidir_weight: float = 0.15            # weight of the reverse-time vote in the harmonic fusion
    secondary_edge_weight: float = 0.15   # max weight of the second-seed edge logits (low-margin only)
    secondary_low_margin: float = 0.35    # primary top1-top2 margin below which the second seed votes
    secondary_mix_mode: str = "low_margin"  # "full": blend (1-w)*primary + w*secondary logits for every pair (two-linker ensemble)
    tertiary_edge_ckpt: str = ""          # third linker: transformer weights that share the PRIMARY U-Net (e.g. a Zebrahub fine-tune); empty = off
    tertiary_edge_weight: float = 0.0     # final logits = (1-w) * (primary/secondary mix) + w * third linker
    tertiary_min_dets: int = 0            # soft density gate: weight 0 below this many detections in the source frame ...
    tertiary_full_dets: int = 0           # ... rising linearly to the full weight at this many (<= min_dets: hard gate at min_dets)
    # Rich cache for offline post-processing experiments (only used with `run --cache-dir`)
    cache_low_thr: float = 0.5            # also keep detection peaks with sigmoid prob above this ("weak" nodes)
    cache_gate_um: float = 15.0           # store transformer logits for node pairs closer than this
    cache_weak_cap: float = 2.0           # at most this many weak nodes per strong node in a frame
    cache_weak_min: int = 100             # but always allow this many weak nodes per frame
    cache_lite: bool = False              # weak peaks and frozen flags only (what the repair fixes need at submission)
    # ---- fixes for what the public pipeline ignores (all off by default = public behaviour) ----
    relink_keep_forks: bool = False       # keep ILP division forks through the motion relink
    relink_flow: bool = False             # predict positions with the neighbour-consensus flow, not own velocity only
    relink_flow_own: float = 0.5          # weight of the own residual velocity on top of the flow
    flow_conf_prob: float = 0.9           # ILP edges with at least this probability define the flow field
    flow_radius_um: float = 25.0
    flow_sigma_um: float = 10.0
    flow_min_neighbors: int = 5           # fewer confident neighbours: fall back to the frame's global drift
    flow_stat: str = "median"             # "mean": Gaussian-weighted mean of the neighbours' displacements instead of the median
    gap_flow: bool = False                # gap closing measures residuals to flow-predicted positions
    gap_use_weak: bool = False            # gap nodes snap to a cached weak peak near the prediction
    rescue_weak: bool = False             # extend track ends/starts into cached weak peaks at predicted positions
    rescue_max_steps: int = 3
    rescue_gate_um: float = 4.0
    rescue_min_prob: float = 0.6
    rescue_clearance_um: float = 4.0      # no other node may be this close to a rescued peak
    rescue_max_frac: float = 0.05         # rescued nodes per video, as a fraction of the graph's nodes
    rescue_dropped: bool = False          # rescue also looks at strong detections the ILP or the filters dropped
    safe_div_flow: bool = False           # daughter gates measured from the parent's flow-predicted position
    safe_div_min_angle_deg: float = 0.0   # daughters must leave the parent in opposite directions (angle >=), 0 = off
    safe_div_cnn_min: float = 0.0         # veto: CNN parent(s) + daughter(c1) + daughter(q) probabilities below this, 0 = off
    safe_div_steal_min: float = 0.0       # >0: a daughter linked to another track may be taken over at this CNN score
    safe_div_dropped_min: float = 0.0     # >0: a dropped strong detection may become the daughter at this CNN score
    safe_div_weak_min: float = 0.0        # >0: a sub-threshold (weak) peak may become the daughter at this CNN score
    safe_div_weak_prob: float = 0.5       # minimum detector probability of such a weak peak
    safe_div_kind_priority: bool = False  # accept orphan proposals before dropped/weak and steal ones within the caps
    safe_div_nodiverge_cnn_min: float = 0.0   # >0: orphans without divergence evidence at t+2 are kept if the CNN sum reaches this
    safe_div_asym_cnn_min: float = 0.0        # >0: orphans failing the symmetry gate are kept if the CNN sum reaches this
    safe_div_traj_cnn_min: float = 0.0        # >0: orphans failing angle/symmetry are kept when the sisters keep separating over t+1..t+3 (CNN bar = this)
    safe_div_traj_margin_um: float = 1.0      # required total growth of the sister distance from t+1 to t+3
    safe_div_dropped_dc_min: float = 0.0      # >0: a dropped strong detection failing the DeepCenter gate may still become the daughter at this CNN sum
    div_min_age: int = 0                      # >0: one lineage cannot fork twice within this many frames (GT: never within a 100-frame clip); one fork is undone
    div_min_age_keep: str = "cnn"             # which of the two forks survives: cnn (higher P(parent)), first (earlier), last (later)
    radius_dir: str = ""                      # lab only: precomputed nucleus radii per cached detection (rad, rad_rel)
    safe_div_size_beta: float = 0.0           # >0: small daughters add beta * clip((size_ref - rad_rel) / size_scale, 0, 1) to the CNN sums
    safe_div_size_ref: float = 0.9            # rad_rel at which the size bonus starts (fresh daughters ~0.65-0.75 of their neighbours)
    safe_div_size_scale: float = 0.25         # rad_rel span over which the bonus grows to beta
    safe_div_size_veto: float = 0.0           # >0: a proposed second daughter with rad_rel above this is rejected
    div_pair_model: str = ""                  # division-pair CNN (pair_cnn.py): scores the (mother, daughter, daughter) triplet as a whole
    div_pair_min: float = 0.5                 # pair probability a proposal needs (modes "and" / "replace")
    div_pair_mode: str = "and"                # and: CNN sums AND pair >= min | or: CNN sums OR pair >= div_pair_or_min | replace: pair only
    div_pair_or_min: float = 0.8
    fork_radial_deg: float = 0.0              # >0: undo forks whose sister vector is within this angle of the local tissue normal (divisions are tangential after 4 hpf)
    fork_radial_radius_um: float = 40.0       # neighbourhood of strong detections used for the local normal (smallest principal axis)
    fork_radial_max_aniso: float = 1.0        # apply only where the normal is defined: sqrt(smallest/middle eigenvalue) below this
    shape_dir: str = ""                       # lab: per-detection nucleus shape (aniso, axis) from shape_cache.py
    safe_div_align_aniso: float = 0.0         # >0: a mother stretched at least this much (sqrt(l1/l2)) ...
    safe_div_align_cos: float = 0.8           # ... whose axis is within acos(cos) of the sister vector lowers the CNN bars below
    safe_div_align_steal_min: float = 1.2     # steal bar for aligned proposals (default 1.8-2.0)
    safe_div_align_orphan_min: float = 0.0    # orphan bar for aligned proposals (default 0.2)
    safe_div_sister_size_max: float = 0.0     # >0: steal also allowed at safe_div_sister_size_steal_min when both daughters are small ...
    safe_div_sister_size_sym: float = 0.25    # ... (rad_rel <= size_max) and alike (|r1 - r2| / mean <= sym)
    safe_div_sister_size_steal_min: float = 1.2
    safe_div_temporal_w: float = 0.0               # >0: CNN sums also count the mother's predecessor P(parent) and the daughters' successors P(daughter), weighted
    safe_div_steal_thief_parent_max: float = 1.0   # <1: no steal from a track whose own node at t looks like a mother (P(parent) above this)
    safe_div_steal_alt_min: float = 0.0            # >0: a steal whose victim has an orphan continuation of its own nearby passes at this CNN bar ...
    safe_div_steal_alt_um: float = 6.0             # ... orphan within this distance of the victim's flow-predicted position; the victim is re-linked to it
    gnn_dir: str = ""                         # lab: per-video graph-trunk scores g for the ILP candidate pairs (link_gnn.py score)
    gnn_weight: float = 0.0                   # >0: ILP edge probability = (1 - w) * transformer p + w * g
    gnn_mode: str = "all"                     # all | lowmargin (blend only for sources whose transformer top1 - top2 < gnn_margin)
    gnn_margin: float = 0.35
    mit_calib: str = ""                       # lab: "quantile" maps every video's CNN logit quantiles onto mit_calib_ref (embryo shift)
    mit_calib_ref: str = ""                   # JSON {"levels": [...percentiles...], "1": [logits], "2": [logits]}
    safe_div_steal_parent_min: float = 0.0    # >0: a steal also needs P(parent of the mother) >= this (with a lower sum bar below)
    safe_div_steal_p_sum_min: float = 1.5     # CNN sum bar for steals that pass the parent-probability floor
    frozen_frames: bool = False           # identical consecutive frames: double the gates of the next transition
    refine_all: bool = False              # intensity-centroid refinement of every node (not only synthetic ones)
    coord_shift_zyx: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # added to every output position (voxels)
    center_model: str = ""                # learned click-offset CNN (center_cnn.py): every detection moves by its vector
    center_offset_dir: str = ""           # lab only: precomputed offset vectors per cached detection
    center_scale: float = 1.0             # fraction of the predicted vector to apply
    center_tta: bool = False              # average the click-offset over y/x flips
    center_stage: str = "post"            # "post": after linefit; "pre": before the motion relink (positions feed the repair)
    short_keep_boundary: bool = False     # components touching the first or last frame are exempt from min_track_len
    stitch: bool = False                  # second pass: track ends -> starts over 1..stitch_max_gap missing frames
    stitch_max_gap: int = 3
    stitch_um_per_step: float = 4.5       # flow residual gate per transition
    stitch_constellation_weight: float = 0.3   # cost weight of the neighbour-constellation mismatch (µm per µm)
    stitch_feature_weight: float = 0.0    # cost weight of (1 - cosine) between cached UNet features, 0 = off
    stitch_k: int = 8                     # neighbours in the constellation descriptor
    stitch_max_frac: float = 0.01         # accepted stitches per video, as a fraction of the edges
    # ---- ILP re-solved from the cache (pp_lab only) ----
    ilp_from_cache: bool = False          # rebuild the ILP input from the cache instead of reading the raw graph
    ilp_use_weak: bool = False            # weak peaks become ILP candidates (with the strong+weak logit regime)
    ilp_node_weight: float = 0.0          # linajea-style node cost w * (det_threshold - det_prob), 0 = off
    ilp_dc_weight: float = 0.0            # node cost w * max(0, tau - DeepCenter score): a doubtful detection must pay to be on a track
    ilp_dc_tau: float = 0.4
    ilp_edge_offset: float = 0.0          # edge weight = ilp_edge_weight * (prob - offset); HOCT uses -1 * (p - 0.5)
    # Adaptive track-end cost: when the linker is unsure on a video (many selected links below `ilp_uncertain_prob`),
    # a fixed disappearance cost of 2.0 drops whole tracks of real cells; re-solve that video with a lower cost.
    ilp_off: bool = False                     # skip the ILP: every strong detection and candidate link goes to repair
    ilp_adaptive_table: list = field(default_factory=list)  # [[uncertainty threshold, disappearance cost], ...]; cost < 0 = no ILP
    ilp_adaptive_disappearance: float = 0.0   # the lower cost for uncertain videos, 0 = off
    ilp_adaptive_uncertainty: float = 0.4     # share of selected links with prob below ilp_uncertain_prob
    ilp_uncertain_prob: float = 0.8
    # Local strength: each node's track-end cost scales with the confidence of its best outgoing link,
    # cost = dmin + (ilp_disappearance - dmin) * conf^gamma; a node with nowhere to go ends cheaply. dmin < 0 = off.
    ilp_local_dmin: float = -1.0
    # Division-aware ILP: per-node division cost lowered for likely parents and appearance cost raised for likely
    # daughters, both from the cell-state CNN (probabilities of the strong detections), so the ILP itself forks.
    ilp_div_cnn: bool = False
    ilp_cnn_views: int = 4                # TTA views of the cell-state CNN when scoring every detection for the ILP costs
    ilp_cnn_model: str = ""               # model(s) for the ILP-cost pass over every detection; empty = div_mitosis_model
    ilp_div_edge_beta: float = 0.0        # edge bonus beta * P(parent of source) * P(daughter of target) before the ILP
    ilp_div_edge_penalty: float = 0.0     # edge penalty beta * P(daughter of target) * (1 - P(parent of source)): continuations into daughters
    mitosis_norm: str = "video"           # "crop": standardise each cell-state crop on its own (model must be trained so)
    v1284_head: str = ""                  # public V1284 head: sub-voxel coordinate refinement from UNet features BEFORE the linker
    ilp_div_gamma: float = 1.0            # division cost = ilp_division - gamma * P(parent)
    ilp_app_gamma: float = 0.0            # appearance cost = ilp_appearance + gamma * P(daughter)
    ilp_div_top2_beta: float = 0.0        # division cost also drops by beta * max(0, p_top2 - tau): the linker's own second child
    ilp_div_top2_tau: float = 0.5         # ... where p_top2 = second-highest candidate probability leaving the node
    ilp_div_track_max: int = 0            # >0: P(parent) for the division cost = max over the node and its k predecessors (first-pass links)
    ilp_rebuild_candidates: bool = False  # lab: rebuild the ILP candidates from the gated logits at edge_threshold (ignore the cached 0.48 set)
    ilp_track_first_thr: float = 0.48     # production ilp_div_track_max: candidate threshold of the first-pass (plain) ILP that gives the predecessors
    ilp_local_gamma: float = 1.0
    ilp_flow_penalty: float = 0.0         # motion-aware re-solve: logit -= penalty * (residual / sigma)^2 / 2, 0 = off
    ilp_flow_sigma_um: float = 3.0        # residual = |displacement - neighbour flow| in µm
    ilp_flow_source: str = "ilp"          # flow for the prior from the first ILP solution, or "candidates" (one ILP only)
    # Learned link re-scorer (LightGBM over linker, motion and raw-crop appearance features); replaces the ILP input
    # probabilities. Needs the rich cache and the frames.
    rescore_model_path: str = ""
    rescore_blend: float = 1.0            # 1 = model logit only; <1 mixes in the transformer's logit
    rescore_temp: float = 1.0             # softmax temperature over each target's parents
    rescore_mode: str = "model"           # "model" = learned re-scorer; "tiebreak" = crop correlation only on near-ties;
                                          # "rankbonus" = mutual-best bonus on the fused logits (public D/E variants)
    rank_beta: float = 0.12               # rankbonus: +beta best parent of the child, +beta/2 best child of the parent, +beta/2 both
    tiebreak_gap: float = 0.3             # tie: best and second parent probabilities closer than this
    tiebreak_min_p2: float = 0.1
    tiebreak_beta: float = 5.0            # weight exp(beta * (ncc - best ncc among the target's parents))
    rescore_device: str = "cpu"
    rescore_feature_dir: str = ""         # lab only: precomputed link features per video
    rescore_drop: list = field(default_factory=list)   # feature names the model was trained without
    edge_threshold: float = 0.48          # candidate edge probability for the ILP (public value)
    edge_temp: float = 1.0                # softmax temperature of the linker logits per target (1 = as inferred; <1 sharper)
    edge_temp_auto: float = 0.0           # >0: per video, sharpen until the median top-1 probability reaches this reference (0 = off)
    edge_temp_min: float = 0.6            # lowest temperature the per-video rule may pick

    # ---- ILP (costs; the edge term is -1 x probability) ----
    ilp_edge_weight: float = -1.0
    ilp_appearance: float = 0.0
    ilp_disappearance: float = 2.0
    ilp_division: float = 1.2

    # ---- graph repair ----
    edge_max_um: float = 14.0
    relink: bool = True
    relink_tight_um: float = 6.0
    relink_relaxed_um: float = 10.0
    relink_dense_per_frame: int = 0       # frames with at least this many nodes use the dense gates below (0 = off)
    relink_dense_tight_um: float = 5.0    # crowded frames: tighter gates cut swaps (worst dense videos +0.005)
    relink_dense_relaxed_um: float = 8.0
    relink_velocity: float = 0.5
    relink_bonus: float = 1.0
    relink_raw_weight: float = 0.05
    relink_max_frame_nodes: int = 2600
    relink_mode: str = "flow"             # "public": the Harmonic Fusion V3 relink verbatim (seed pass -> k-median flow -> tight/relaxed)
    pub_seed_gate_um: float = 5.5         # public MOTION_RELINK_TIGHT_UM (seed pass and flow-less frames)
    pub_tight_um: float = 7.0             # public MOTION_RELINK_FLOW_TIGHT_UM
    pub_flow_k: int = 12                  # public MOTION_RELINK_FLOW_K
    pub_flow_radius_um: float = 40.0      # public MOTION_RELINK_FLOW_RADIUS_UM
    pub_flow_exclude_um: float = 1.5      # public MOTION_RELINK_FLOW_EXCLUDE_UM
    pub_flow_min_samples: int = 4         # public MOTION_RELINK_FLOW_MIN_SAMPLES

    gap_close: bool = True
    gap_um: float = 5.0                   # per step; end at t -> start at t+2 within 2 x gap_um
    gap_density_adaptive: bool = False    # public: True
    gap_density_reference_um: float = 6.5
    gap_density_gain: float = 0.040
    gap_density_max_step_delta_um: float = 0.125
    gap_density_neighbors: int = 3
    gap_reuse_um: float = 3.2
    gap_max_added_frac: float = 0.05
    gap_max_added_abs: int = 2000
    refine_synthetic: bool = True
    refine_win_z: int = 1
    refine_win_yx: int = 3
    refine_max_shift_um: float = 3.2

    gap2: bool = True
    gap2_max_total_um: float = 10.2
    gap2_max_step_um: float = 4.4
    gap2_max_links_frac: float = 0.0045
    gap2_max_links_abs: int = 180
    gap2_frame_frac_cap: float = 0.006

    safe_div: bool = True
    safe_div_parent_um: float = 9.0
    safe_div_sister_um: float = 14.0
    safe_div_existing_child_um: float = 10.0
    safe_div_symmetry_tau: float = 0.6
    safe_div_diverge_um: float = 2.25
    safe_div_horizon_frames: int = 0      # >0: divergence may show up within this many frames after the sisters (public "horizon")
    safe_div_horizon_um: float = 2.8      # ... or the sisters' descendants are at least this far apart within the horizon
    leaf_prune_min_prob: float = 0.0      # >0: drop a terminal node (no successor, single parent, not last frame) whose only edge has p below this
    safe_div_mutual_nn: bool = True
    safe_div_frame_frac_cap: float = 0.0076
    safe_div_global_frac_cap: float = 0.00375
    # Learned divisions: a LightGBM model scores every fork candidate (parent with one child, second child near the
    # parent's flow-predicted position) and replaces the hand-made gates. Runs before safe_divisions.
    div_learn: bool = False
    div_model_path: str = ""              # LightGBM text model; empty = candidates are only dumped (training data)
    div_threshold: float = 0.5
    div_steal: bool = False               # a candidate daughter linked to another track may be taken over
    div_steal_threshold: float = 0.8
    div_radius_um: float = 14.0           # second-daughter search radius around the flow-predicted parent position
    div_dump_dir: str = ""                # write candidate features + graph nodes per video (training data)
    div_min_angle_deg: float = 60.0       # candidate prefilter: angle between the two daughters seen from the parent
    div_ncc: bool = False                 # raw-crop cross-correlations as features (slow; needs frames)
    div_use_dropped: bool = False         # strong detections missing from the graph may become the second daughter
    div_mitosis_model: str = ""           # 3D cell-state CNN (normal / parent / daughter) scored on the fly
    div_mitosis_dir: str = ""             # lab only: precomputed cell-state scores of every cached detection

    deepcenter: bool = True
    deepcenter_views: str = "d4_legacy"   # TTA views for the DeepCenter heatmap ("none" = single view)
    deepcenter_gap_threshold: float = 0.25
    deepcenter_gap_min_span_um: float = 8.5
    deepcenter_safe_div_threshold: float = 0.25
    deepcenter_win_z: int = 1
    deepcenter_win_yx: int = 2

    prune_isolated: bool = True
    min_track_len: int = 6
    keep_division_components: bool = True
    short_track_rescue: bool = False      # public: True
    rescue_trigger_removed_frac: float = 0.10
    rescue_min_len: int = 4
    rescue_min_mean_prob: float = 0.88
    rescue_max_mean_dist_um: float = 3.0
    rescue_max_nodes_frac: float = 0.012
    rescue_max_nodes_abs: int = 120

    linefit_weight: float = 0.8
    linefit_window: int = 2

    @classmethod
    def load(cls, source: str | dict | None = None, **overrides) -> "Config":
        data: dict = {}
        if isinstance(source, dict):
            data.update(source)
        elif source:
            text = Path(source).read_text()
            data.update(json.loads(text))
        preset = data.pop("preset", None)
        if preset:
            data = {**PRESETS[preset], **data}
        data.update(overrides)
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)


# "public_reference" reproduces the public reference notebook #5 (flexonafft/biohub-harmonic-fusion, base
# configuration before its PP sweep) graph for graph. "clean" (the defaults) keeps every model-level setting of
# the reference and drops the two repair micro-features that had no measurable effect (identical official score
# on 8 train videos with and without them). Model-level changes (TTA views, knob values) wait for held-out
# validation: see README.md.
PRESETS: dict[str, dict] = {
    "clean": {},
    "public_reference": {
        "gap_density_adaptive": True,
        "short_track_rescue": True,
    },
}


# =============================================================================================
# Frame access
# =============================================================================================

_ZARR_META: dict[str, tuple[tuple[int, ...], np.dtype]] = {}


def zarr_meta(zarr_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    key = str(zarr_path)
    if key not in _ZARR_META:
        meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
        _ZARR_META[key] = (tuple(int(v) for v in meta["shape"]), np.dtype(meta["data_type"]))
    return _ZARR_META[key]


def load_frame_phase(zarr_arr, t: int, target_shape, downsample, phase) -> "torch.Tensor":
    """The support pack's strided subsampling starting at a shifted grid phase (pz, py, px) in raw voxels; padded at the
    far end so the tensor has the phase-0 shape (pixel k of this grid sits at raw position k*d + phase)."""
    import torch
    import torch.nn.functional as F
    dz, dy, dx = downsample
    pz, py, px = phase
    raw = zarr_arr[t, pz::dz, py::dy, px::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    pad = [0, max(0, int(target_shape[2]) - frame.shape[2]), 0, max(0, int(target_shape[1]) - frame.shape[1]),
           0, max(0, int(target_shape[0]) - frame.shape[0])]
    if any(pad):
        frame = F.pad(frame[None, None], pad, mode="replicate")[0, 0]
    return frame[:int(target_shape[0]), :int(target_shape[1]), :int(target_shape[2])].contiguous()


def read_frame(zarr_path: Path, t: int) -> np.ndarray:
    """One (Z, Y, X) frame, decoded straight from its blosc chunk (zarr fallback)."""
    shape, dtype = zarr_meta(zarr_path)
    chunk = zarr_path / "0" / "c" / str(int(t)) / "0" / "0" / "0"
    try:
        import blosc2
        arr = np.frombuffer(blosc2.decompress(chunk.read_bytes()), dtype=dtype)
        if arr.size == int(np.prod(shape[1:])):
            return arr.reshape(shape[1:]).copy()
    except Exception:
        pass
    import zarr
    return np.asarray(zarr.open(str(zarr_path / "0"), mode="r")[int(t)])


class FrameCache:
    def __init__(self, zarr_path: Path, max_frames: int = 16):
        self.zarr_path = zarr_path
        self.max_frames = max_frames
        self._frames: dict[int, np.ndarray] = {}

    def get(self, t: int) -> np.ndarray:
        t = int(t)
        if t not in self._frames:
            self._frames[t] = read_frame(self.zarr_path, t)
            while len(self._frames) > self.max_frames:
                self._frames.pop(next(iter(self._frames)))
        return self._frames[t]


# =============================================================================================
# Geometry helpers (all distances in microns)
# =============================================================================================

def point_um(node: dict) -> np.ndarray:
    return np.array([float(node["z"]) * SCALE_UM[0], float(node["y"]) * SCALE_UM[1],
                     float(node["x"]) * SCALE_UM[2]], dtype=np.float64)


def dist_um(a: dict, b: dict) -> float:
    dz = (float(a["z"]) - float(b["z"])) * SCALE_UM[0]
    dy = (float(a["y"]) - float(b["y"])) * SCALE_UM[1]
    dx = (float(a["x"]) - float(b["x"])) * SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def dist_points_um(a: tuple, b: tuple) -> float:
    dz = (a[0] - b[0]) * SCALE_UM[0]
    dy = (a[1] - b[1]) * SCALE_UM[1]
    dx = (a[2] - b[2]) * SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def zyx(node: dict) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_rank_key(edge: dict) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    return (float(prob) if prob is not None else 0.0), -float(edge["distance_um"])


# =============================================================================================
# TTA views on the last two axes (Y, X)
# =============================================================================================

TTA_VIEWS = {
    "d4": ["id", "fx", "fy", "fxy", "r90", "r270", "t", "at"],
    # public notebooks: their "anti-transpose" was rot90(k=1) then transpose, which is exactly the x-flip
    "d4_legacy": ["id", "fx", "fy", "fxy", "r90", "r270", "t", "fx"],
    "d4_7": ["id", "fx", "fy", "fxy", "r90", "r270", "t"],  # the 7 distinct public views, no duplicate
    "flips4": ["id", "fx", "fy", "fxy"],
    # the model's own training augmentation group: independent flips of Z, Y and X (flip_augment)
    "flips8": ["id", "fx", "fy", "fxy", "fz", "fzx", "fzy", "fzyx"],
    "none": ["id"],
}


def view_fwd(x, v: str):
    import torch
    if v == "id":
        return x
    if v == "fx":
        return x.flip(-1)
    if v == "fy":
        return x.flip(-2)
    if v == "fxy":
        return x.flip(-2, -1)
    if v == "r90":
        return torch.rot90(x, 1, dims=(-2, -1))
    if v == "r270":
        return torch.rot90(x, 3, dims=(-2, -1))
    if v == "t":
        return x.transpose(-1, -2)
    if v == "at":  # reflection across the anti-diagonal: out[i, j] = in[W-1-j, H-1-i]
        return x.transpose(-1, -2).flip(-2, -1)
    if v == "fz":
        return x.flip(-3)
    if v == "fzx":
        return x.flip(-3, -1)
    if v == "fzy":
        return x.flip(-3, -2)
    if v == "fzyx":
        return x.flip(-3, -2, -1)
    raise ValueError(v)


def view_inv(x, v: str):
    import torch
    if v == "r90":
        return torch.rot90(x, -1, dims=(-2, -1))
    if v == "r270":
        return torch.rot90(x, -3, dims=(-2, -1))
    return view_fwd(x, v)  # every other view is an involution


# =============================================================================================
# Models
# =============================================================================================

def import_support_pack(pack: Path):
    repo = Path(pack) / "repo"
    for sub in ("src", "scripts"):
        path = str(repo / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import predict_unet_transformer as support  # the unmodified support-pack module
    return support


def weights_path(pack: Path) -> Path:
    return Path(pack) / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"


class TrackingModels:
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device
        self.support = import_support_pack(Path(cfg.support_pack))
        self.primary, self.window, self.downsample = self.support.load_model(weights_path(cfg.support_pack), device)
        self.secondary = None
        if cfg.use_secondary:
            self.secondary, w2, ds2 = self.support.load_model(weights_path(cfg.secondary_pack), device)
            if (w2, tuple(ds2)) != (self.window, tuple(self.downsample)):
                raise ValueError("primary and secondary models use different windows/downsampling")
        self.tertiary = None
        if cfg.tertiary_edge_ckpt and cfg.tertiary_edge_weight > 0:
            self.tertiary = load_tertiary(self.primary, cfg.tertiary_edge_ckpt, device)


def load_tertiary(primary, ckpt: str, device):
    """A copy of the primary model with the third linker's weights; its U-Net must equal the primary's, because the edge
    features are taken from the primary encoder."""
    import copy
    import torch
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    model = copy.deepcopy(primary)
    ref = primary.state_dict()
    model.load_state_dict(obj, strict=True)
    unet = [k for k in obj if k.startswith("unet.")]
    same = sum(bool(torch.equal(obj[k].to(ref[k].device), ref[k])) for k in unet)
    if not unet or same != len(unet):
        raise ValueError(f"third linker U-Net differs from the primary ({same}/{len(unet)} tensors equal)")
    print(f"third linker loaded from {ckpt}: U-Net identical to the primary ({same} tensors)", flush=True)
    return model.to(device).eval()


def tertiary_weight(cfg, n_src: int) -> float:
    """Soft density gate for the third linker: 0 below tertiary_min_dets, linear up to the full weight at tertiary_full_dets."""
    w = cfg.tertiary_edge_weight
    if n_src < cfg.tertiary_min_dets:
        return 0.0
    if cfg.tertiary_full_dets > cfg.tertiary_min_dets:
        return w * min(1.0, (n_src - cfg.tertiary_min_dets) / (cfg.tertiary_full_dets - cfg.tertiary_min_dets))
    return w


def tertiary_mix(models, cfg, logits, args_fwd, args_rev, n_src: int):
    """Blend the third linker into the final logits on dense frames."""
    w = tertiary_weight(cfg, n_src) if models.tertiary is not None else 0.0
    if w <= 0:
        return logits
    t_logits = models.tertiary.predict_edges(*args_fwd)
    if cfg.bidir_weight > 0:
        t_logits = harmonic_bidirectional(models.tertiary, t_logits, args_fwd, args_rev, cfg.bidir_weight)
    return (1.0 - w) * logits + w * t_logits


def encode_tta(model, imgs, views: list[str]):
    """Returns (identity-view features, view-averaged features, view-averaged detection logits)."""
    feat_id, det_id = model.encode(imgs)
    det_acc = list(det_id)
    feat_acc = feat_id.clone()
    for v in views[1:]:
        feat_v, det_v = model.encode(view_fwd(imgs, v))
        for f in range(len(det_acc)):
            det_acc[f] = det_acc[f] + view_inv(det_v[f], v)
        feat_acc = feat_acc + view_inv(feat_v, v)
        del feat_v, det_v
    n = len(views)
    return feat_id, feat_acc / n, [d / n for d in det_acc]


def align_to(reference, other, dim=None):
    """Affine-align `other` to the mean/std of `reference` (std ratio clamped to [0.5, 2])."""
    if dim is None:
        ref_c, oth_c = reference.mean(), other.mean()
        ref_s = reference.float().std(unbiased=False).clamp_min(1e-4)
        oth_s = other.float().std(unbiased=False).clamp_min(1e-4)
    else:
        ref_c, oth_c = reference.mean(dim=dim, keepdim=True), other.mean(dim=dim, keepdim=True)
        ref_s = reference.float().std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-4)
        oth_s = other.float().std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-4)
    return (other - oth_c) * (ref_s / oth_s).clamp(0.5, 2.0) + ref_c


def harmonic_bidirectional(model, logits, args_fwd, args_rev, weight: float):
    """Fuse forward logits (1, n_src, n_tgt) with the reverse-time vote in probability space."""
    import torch
    reverse = model.predict_edges(*args_rev).transpose(1, 2)
    fwd_c = logits.mean(dim=1, keepdim=True)
    fwd_s = logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    rev_c = reverse.mean(dim=1, keepdim=True)
    rev_s = reverse.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    reverse = (reverse - rev_c) * (fwd_s / rev_s).clamp(0.5, 2.0).to(reverse.dtype) + fwd_c
    p_f = torch.softmax(logits.float(), dim=1).clamp_min(1e-8)
    p_r = torch.softmax(reverse.float(), dim=1).clamp_min(1e-8)
    p_h = 1.0 / ((1.0 - weight) / p_f + weight / p_r)
    p_h = p_h / p_h.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_h = torch.log(p_h.clamp_min(1e-8))
    h_c = log_h.mean(dim=1, keepdim=True)
    h_s = log_h.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((log_h - h_c) * (fwd_s / h_s).clamp(0.5, 2.0) + fwd_c).to(reverse.dtype)


def secondary_low_margin_mix(logits, secondary_logits, weight: float, low_margin: float, mode: str = "low_margin"):
    """Blend second-seed logits only where the primary is unsure and both seeds agree on the top parent
    (mode "full": plain logit blend for every pair)."""
    import torch
    aligned = align_to(logits, secondary_logits, dim=1)
    if mode == "full":
        return (1.0 - weight) * logits + weight * aligned
    n_src = logits.shape[1]
    if n_src < 2:
        return logits
    p_prim = torch.softmax(logits[0], dim=0)
    p_sec = torch.softmax(aligned[0], dim=0)
    top_p = torch.topk(p_prim, k=2, dim=0)
    top_s = torch.topk(p_sec, k=2, dim=0)
    margin = top_p.values[0] - top_p.values[1]
    uncertainty = ((low_margin - margin) / low_margin).clamp(0.0, 1.0)
    local = torch.where(top_p.indices[0].eq(top_s.indices[0]), weight * uncertainty, torch.zeros_like(margin))
    blend = local.view(1, 1, -1)
    return (1.0 - blend) * logits + blend * aligned


V1284_OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def v1284_load(path: str, device):
    """The public coordinate head (anvithpothula/biohub-v1284-head-s075): Linear(224, 32) - SiLU - Linear(32, 3)."""
    import torch
    import torch.nn as nn
    key = ("v1284", path, str(device))
    if key not in _DIV_MODELS:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        head = nn.Sequential(nn.Linear(224, 32), nn.SiLU(), nn.Linear(32, 3)).to(device)
        head.load_state_dict(saved["state_dict"])
        _DIV_MODELS[key] = (head.eval(), saved["mean"].to(device), saved["scale"].to(device))
    return _DIV_MODELS[key]


def v1284_refine(arr: np.ndarray, feature, head_pack, voxel_um, max_um: float = 2.0) -> np.ndarray:
    """Detections (n, 4) [t, z, y, x] on the downsampled grid -> float coordinates moved by the head (<= max_um)."""
    import torch
    if not len(arr):
        return arr.astype(np.float32)
    head, mean, scale = head_pack
    dev = feature.device
    xyz = torch.as_tensor(np.rint(arr[:, 1:]).astype(np.int64), device=dev)
    blocks = []
    for off in V1284_OFFSETS:
        loc = xyz + torch.tensor(off, device=dev)
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size - 1)
        blocks.append(feature[0, :, loc[:, 0], loc[:, 1], loc[:, 2]].T)
    x = torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1).float()
    delta = head((x - mean) / scale)
    delta = max_um * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))      # bounded, in µm
    shift = (delta / torch.as_tensor(voxel_um, device=dev, dtype=torch.float32)).cpu().numpy()
    out = arr.astype(np.float32).copy()
    out[:, 1:] += shift
    out[:, 1:] = np.clip(out[:, 1:], 0, np.asarray(feature.shape[-3:], dtype=np.float32) - 1)
    return out


def trilinear_features(maps, coords, mask):
    """(1, C, Z, Y, X) sampled at fractional coordinates (1, n, 3) -> (1, n, C); integer coordinates = native gather."""
    import torch
    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)
    for b in range(len(maps)):
        n = int(mask[b].sum())
        if not n:
            continue
        q = coords[b, :n].clone()
        for axis, size in enumerate(maps.shape[-3:]):
            q[:, axis].clamp_(0, size - 1)
        low = q.floor().long()
        frac = q - low
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    shift = torch.tensor([z, y, x], device=maps.device)
                    loc = low + shift
                    for axis, size in enumerate(maps.shape[-3:]):
                        loc[:, axis].clamp_(0, size - 1)
                    w = torch.where(shift.bool(), frac, 1 - frac).prod(dim=1)
                    out[b, :n] += maps[b, :, loc[:, 0], loc[:, 1], loc[:, 2]].T * w[:, None]
    return out


def predict_video(zarr_path: Path, models: TrackingModels, cfg: Config, cache: dict | None = None):
    """Detections (N, 4) [t, z, y, x] in original voxels and candidate edges (i, j, prob, dist).

    With `cache` (a dict) the function also fills it with the material for offline experiments: weak
    detection peaks with their scores, transformer logits for all node pairs within `cache_gate_um`
    (strong-only regime as used by the baseline, and the strong+weak regime), the log-sum-exp per target
    needed to turn logits into the pipeline's softmax probabilities, per-node UNet features and
    frozen-frame flags. The baseline outputs are unchanged by the cache.
    """
    import torch
    import zarr
    sp = models.support
    device = models.device
    W, downsample = models.window, tuple(models.downsample)
    views = TTA_VIEWS[cfg.tta]
    if views[0] != "id":
        raise ValueError("the first TTA view must be the identity")

    ds = sp.open_dataset(zarr_path, normalize=False, load_image=False, downsample=downsample)
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    T = ds.image_shape[0]
    image_shape = (T,) + tuple(ds.image_shape[1:])
    target_shape = list(image_shape[1:])
    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = sp.pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)
    window_shape = (W,) + image_shape[1:]

    starts = list(range(0, T - W + 1, max(W - 1, 1)))
    if not starts or starts[-1] + W < T:
        last = max(T - W, 0)
        if not starts or last != starts[-1]:
            starts.append(last)

    coord_lists: list[np.ndarray] = []
    offsets: dict[int, tuple[int, int]] = {}
    n_total = 0
    seen_pairs: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, float, float]] = []
    guard_fallbacks = 0
    n_phase_added = 0

    # ---- rich cache state (see the docstring) ----
    vs_t = torch.tensor(voxel_size, dtype=torch.float32, device=device)  # µm per downsampled voxel
    strong_prob_lists: list[np.ndarray] = []
    weak_lists: list[np.ndarray] = []            # (n_w, 4) [t, z, y, x] downsampled grid, per frame
    weak_prob_lists: list[np.ndarray] = []
    weak_offsets: dict[int, tuple[int, int]] = {}
    n_weak_total = 0
    feat_store: dict[int, np.ndarray] = {}       # t -> (n_strong + n_weak, C) float16, strong rows first
    ext_pairs: list[np.ndarray] = []             # (m, 3) [src_code, tgt_code, logit]; weak code = -(idx + 1)
    ext_lse: list[np.ndarray] = []               # (n_tgt_all, 3) [tgt_code, lse_all_parents, lse_strong_parents]
    base_pairs: list[np.ndarray] = []            # (m, 3) [src_id, tgt_id, logit] strong-only regime
    base_lse: list[np.ndarray] = []              # (n_tgt, 2) [tgt_id, lse_strong_parents]
    frozen = np.zeros(max(T - 1, 0), dtype=bool)

    head_pack = v1284_load(cfg.v1284_head, device) if cfg.v1284_head else None

    def index_features(model, maps, xyz, m):
        return trilinear_features(maps, xyz, m) if head_pack is not None else model._index_features(maps, xyz, m)

    def weak_code(w0: int, k: int) -> int:
        return -(w0 + k) - 1

    def node_set(t: int, f: int):
        """All cached nodes of frame t as model inputs: (coords_ds (n, 3) float, codes (n,), n_strong)."""
        s0, s1 = offsets[t]
        w0, w1 = weak_offsets.get(t, (0, 0))
        c_s = coords_so_far[s0:s1, 1:].astype(np.float32)
        c_w = weak_so_far[w0:w1, 1:].astype(np.float32) if w1 > w0 else np.empty((0, 3), dtype=np.float32)
        codes = np.concatenate([np.arange(s0, s1), np.array([weak_code(w0, k) for k in range(w1 - w0)], dtype=np.int64)])
        return np.concatenate([c_s, c_w]), codes.astype(np.int64), s1 - s0

    def model_inputs(c: np.ndarray, f: int):
        n = len(c)
        xyz = torch.from_numpy(c).unsqueeze(0).to(device)
        rel = np.concatenate([np.full((n, 1), f, dtype=np.float32), c], axis=1)
        pos = torch.from_numpy(sp.extract_pos_features(rel, window_shape)).unsqueeze(0).to(device)
        m = torch.ones(1, n, dtype=torch.bool, device=device)
        return xyz, pos, m

    def cache_pair(f: int, t_src: int, t_tgt: int, feat_p, feat_s, base_logits, s0: int, g0: int, ext: bool = True):
        """Store gated logits for the baseline regime and (unless ext=False) the strong+weak regime of one frame pair."""
        # baseline regime: the exact logits the pipeline turned into probabilities
        L = base_logits[0].float()
        src_um = torch.from_numpy(coords_so_far[s0:s0 + L.shape[0], 1:].astype(np.float32)).to(device) * vs_t
        tgt_um = torch.from_numpy(coords_so_far[g0:g0 + L.shape[1], 1:].astype(np.float32)).to(device) * vs_t
        ii, jj = torch.nonzero(torch.cdist(src_um, tgt_um) <= cfg.cache_gate_um, as_tuple=True)
        base_pairs.append(np.stack([(s0 + ii).cpu().numpy(), (g0 + jj).cpu().numpy(),
                                    L[ii, jj].cpu().numpy()], axis=1).astype(np.float64))
        base_lse.append(np.stack([np.arange(g0, g0 + L.shape[1], dtype=np.float64),
                                  torch.logsumexp(L, dim=0).cpu().numpy()], axis=1))
        if not ext:
            return
        # strong + weak regime
        c_src, code_src, ns_src = node_set(t_src, f)
        c_tgt, code_tgt, ns_tgt = node_set(t_tgt, f + 1)
        if len(c_src) == 0 or len(c_tgt) == 0:
            return
        xyz_s, pos_s, m_s = model_inputs(c_src, f)
        xyz_t, pos_t, m_t = model_inputs(c_tgt, f + 1)
        fs = index_features(models.primary, feat_p[:, f], xyz_s, m_s)
        ft = index_features(models.primary, feat_p[:, f + 1], xyz_t, m_t)
        a_fwd = (fs, ft, xyz_s * ds_arr_t, xyz_t * ds_arr_t, pos_s, pos_t, m_s, m_t)
        a_rev = (ft, fs, xyz_t * ds_arr_t, xyz_s * ds_arr_t, pos_t, pos_s, m_t, m_s)
        logits = models.primary.predict_edges(*a_fwd)
        if cfg.bidir_weight > 0:
            logits = harmonic_bidirectional(models.primary, logits, a_fwd, a_rev, cfg.bidir_weight)
        if feat_s is not None and cfg.secondary_edge_weight > 0:
            sfs = index_features(models.secondary, feat_s[:, f], xyz_s, m_s)
            sft = index_features(models.secondary, feat_s[:, f + 1], xyz_t, m_t)
            s_logits = models.secondary.predict_edges(sfs, sft, *a_fwd[2:])
            logits = secondary_low_margin_mix(logits, s_logits, cfg.secondary_edge_weight, cfg.secondary_low_margin, cfg.secondary_mix_mode)
        logits = tertiary_mix(models, cfg, logits, a_fwd, a_rev, int(ns_src))
        L = logits[0].float()
        lse_all = torch.logsumexp(L, dim=0)
        lse_strong = torch.logsumexp(L[:ns_src], dim=0) if ns_src > 0 else torch.full_like(lse_all, float("nan"))
        ext_lse.append(np.stack([code_tgt.astype(np.float64), lse_all.cpu().numpy(), lse_strong.cpu().numpy()], axis=1))
        src_um = xyz_s[0] * vs_t
        tgt_um = xyz_t[0] * vs_t
        ii, jj = torch.nonzero(torch.cdist(src_um, tgt_um) <= cfg.cache_gate_um, as_tuple=True)
        ii_np, jj_np = ii.cpu().numpy(), jj.cpu().numpy()
        ext_pairs.append(np.stack([code_src[ii_np].astype(np.float64), code_tgt[jj_np].astype(np.float64),
                                   L[ii, jj].cpu().numpy()], axis=1))
        feat_store[t_src] = fs[0].detach().cpu().numpy().astype(np.float16)      # src-side features win
        if t_tgt not in feat_store:
            feat_store[t_tgt] = ft[0].detach().cpu().numpy().astype(np.float16)

    with torch.no_grad():
        for ws in starts:
            frames = list(range(ws, ws + W))
            imgs = torch.stack([sp._load_frame(zarr_arr, t, target_shape, downsample) for t in frames])
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
            if cache is not None:
                for f in range(W - 1):
                    if frames[f] < T - 1 and torch.equal(imgs[0, f], imgs[0, f + 1]):
                        frozen[frames[f]] = True

            feat_id, feat_mean, det = encode_tta(models.primary, imgs, views)
            feat_p = feat_mean if cfg.edge_feature_tta else feat_id
            del feat_id, feat_mean

            feat_s = None
            if models.secondary is not None:
                s_id, s_mean, s_det = encode_tta(models.secondary, imgs, views)
                w_ft = cfg.secondary_feature_tta_weight if cfg.edge_feature_tta else 0.0
                feat_s = (1.0 - w_ft) * s_id + w_ft * s_mean if w_ft > 0 else s_id
                del s_id, s_mean
                if cfg.secondary_det_weight > 0:
                    for f in range(W):
                        blended = ((1.0 - cfg.secondary_det_weight) * det[f]
                                   + cfg.secondary_det_weight * align_to(det[f], s_det[f]))
                        if cfg.det_retention_guard > 0:
                            n_prim = len(sp._detect_cells_pooled(det[f][0], frames[f], cfg.det_threshold, pool_k))
                            n_blend = len(sp._detect_cells_pooled(blended[0], frames[f], cfg.det_threshold, pool_k))
                            if n_prim > 0 and n_blend / n_prim < cfg.det_retention_guard:
                                guard_fallbacks += int(frames[f] not in offsets)
                                continue
                        det[f] = blended
                del s_det
            del imgs

            phase_extra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if cfg.det_phases > 1:
                # detection on phase-shifted strided grids (the loader is strided, so every phase is a training-consistent
                # view); peaks the phase-0 grid missed are added below (single view, primary detector only)
                from scipy.spatial import cKDTree
                phases = [(0, 2, 2)] if cfg.det_phases == 2 else [(0, 2, 0), (0, 0, 2), (0, 2, 2)]
                for ph in phases:
                    imgs_p = torch.stack([load_frame_phase(zarr_arr, t, target_shape, downsample, ph) for t in frames])
                    imgs_p = ((imgs_p - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
                    _, _, det_p = encode_tta(models.primary, imgs_p, TTA_VIEWS["none"])
                    del imgs_p
                    shift = np.array([ph[0] / downsample[0], ph[1] / downsample[1], ph[2] / downsample[2]], dtype=np.float32)
                    for f, t in enumerate(frames):
                        if t in offsets:
                            continue
                        arr_p = sp._detect_cells_pooled(det_p[f][0], t, cfg.det_phase_threshold or cfg.det_threshold, pool_k)
                        if not len(arr_p):
                            continue
                        vol_p = det_p[f][0, 0]
                        zz, yy, xx = [torch.from_numpy(np.rint(arr_p[:, k]).astype(np.int64)).to(device) for k in (1, 2, 3)]
                        pr = torch.sigmoid(vol_p[zz, yy, xx]).cpu().numpy().astype(np.float32)
                        arr_p = arr_p.astype(np.float32)
                        arr_p[:, 1:] += shift
                        prev = phase_extra.get(f)
                        phase_extra[f] = (arr_p, pr) if prev is None else (np.concatenate([prev[0], arr_p]),
                                                                            np.concatenate([prev[1], pr]))
                    del det_p

            for f, t in enumerate(frames):
                if t not in offsets:
                    arr = sp._detect_cells_pooled(det[f][0], t, cfg.det_threshold, pool_k)
                    if head_pack is not None:
                        arr = v1284_refine(arr, feat_p[:, f], head_pack, voxel_size)
                    extra_prob = None
                    if f in phase_extra:
                        arr_p, pr = phase_extra[f]
                        vs_np = np.array(voxel_size, dtype=np.float64)
                        ump = arr_p[:, 1:].astype(np.float64) * vs_np
                        merge_um = cfg.det_phase_merge_um or cfg.pool_kernel_um
                        keep = np.ones(len(arr_p), dtype=bool)
                        if len(arr):
                            d_near, _ = cKDTree(arr[:, 1:].astype(np.float64) * vs_np).query(ump, k=1)
                            keep &= d_near > merge_um
                        sel = np.nonzero(keep)[0]
                        chosen: list[int] = []
                        for i in sel[np.argsort(-pr[sel])]:          # greedy suppression among the phase peaks
                            if all(np.linalg.norm(ump[i] - ump[j]) > merge_um for j in chosen):
                                chosen.append(int(i))
                        if chosen:
                            n_phase_added += len(chosen)
                            arr = np.concatenate([arr.astype(np.float32), arr_p[chosen]])
                            extra_prob = pr[chosen]
                    offsets[t] = (n_total, n_total + len(arr))
                    n_total += len(arr)
                    coord_lists.append(arr)
                    if cache is not None:
                        vol = det[f][0, 0]
                        n_own = len(arr) - (len(extra_prob) if extra_prob is not None else 0)
                        zz, yy, xx = [torch.from_numpy(np.rint(arr[:n_own, k]).astype(np.int64)).to(device) for k in (1, 2, 3)]
                        p_own = torch.sigmoid(vol[zz, yy, xx]).cpu().numpy().astype(np.float32)
                        strong_prob_lists.append(p_own if extra_prob is None else np.concatenate([p_own, extra_prob]))
                        low = sp._detect_cells_pooled(det[f][0], t, cfg.cache_low_thr, pool_k)
                        lp = np.empty(0, dtype=np.float32)
                        if len(low):
                            lz, ly, lx = [torch.from_numpy(low[:, k].astype(np.int64)).to(device) for k in (1, 2, 3)]
                            lp = torch.sigmoid(vol[lz, ly, lx]).cpu().numpy().astype(np.float32)
                            keep = lp <= cfg.det_threshold            # strong peaks are already in `arr`
                            low, lp = low[keep], lp[keep]
                            cap = max(cfg.cache_weak_min, int(cfg.cache_weak_cap * len(arr)))
                            if len(low) > cap:
                                top = np.argsort(-lp)[:cap]
                                low, lp = low[top], lp[top]
                        weak_offsets[t] = (n_weak_total, n_weak_total + len(low))
                        n_weak_total += len(low)
                        weak_lists.append(low.astype(np.int16))
                        weak_prob_lists.append(lp)
            coords_so_far = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
            weak_so_far = np.concatenate(weak_lists) if weak_lists else np.empty((0, 4), dtype=np.int16)

            for f in range(W - 1):
                t_src, t_tgt = frames[f], frames[f + 1]
                if (t_src, t_tgt) in seen_pairs:
                    continue
                seen_pairs.add((t_src, t_tgt))
                s0, s1 = offsets[t_src]
                g0, g1 = offsets[t_tgt]
                if s1 == s0 or g1 == g0:
                    continue
                c_src, c_tgt = coords_so_far[s0:s1], coords_so_far[g0:g1]
                xyz_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                xyz_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                rel_src, rel_tgt = c_src.copy(), c_tgt.copy()
                rel_src[:, 0], rel_tgt[:, 0] = f, f + 1
                pos_src = torch.from_numpy(sp.extract_pos_features(rel_src, window_shape)).unsqueeze(0).to(device)
                pos_tgt = torch.from_numpy(sp.extract_pos_features(rel_tgt, window_shape)).unsqueeze(0).to(device)
                m_src = torch.ones(1, len(c_src), dtype=torch.bool, device=device)
                m_tgt = torch.ones(1, len(c_tgt), dtype=torch.bool, device=device)

                fs = index_features(models.primary, feat_p[:, f], xyz_src, m_src)
                ft = index_features(models.primary, feat_p[:, f + 1], xyz_tgt, m_tgt)
                args_fwd = (fs, ft, xyz_src * ds_arr_t, xyz_tgt * ds_arr_t, pos_src, pos_tgt, m_src, m_tgt)
                args_rev = (ft, fs, xyz_tgt * ds_arr_t, xyz_src * ds_arr_t, pos_tgt, pos_src, m_tgt, m_src)
                logits = models.primary.predict_edges(*args_fwd)  # (1, n_src, n_tgt)
                if cfg.bidir_weight > 0:
                    logits = harmonic_bidirectional(models.primary, logits, args_fwd, args_rev, cfg.bidir_weight)
                if feat_s is not None and cfg.secondary_edge_weight > 0:
                    sfs = index_features(models.secondary, feat_s[:, f], xyz_src, m_src)
                    sft = index_features(models.secondary, feat_s[:, f + 1], xyz_tgt, m_tgt)
                    s_logits = models.secondary.predict_edges(sfs, sft, *args_fwd[2:])
                    logits = secondary_low_margin_mix(logits, s_logits, cfg.secondary_edge_weight,
                                                      cfg.secondary_low_margin, cfg.secondary_mix_mode)
                logits = tertiary_mix(models, cfg, logits, args_fwd, args_rev, len(c_src))
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()  # normalised over parents
                ii, jj = np.nonzero(probs > cfg.edge_threshold)
                order = sorted(zip(probs[ii, jj].tolist(), ii.tolist(), jj.tolist()), reverse=True)
                for prob, i, j in order:
                    gi, gj = s0 + i, g0 + j
                    d = float(np.linalg.norm(coords_so_far[gi, 1:].astype(np.float32)
                                             - coords_so_far[gj, 1:].astype(np.float32)))
                    edges.append((gi, gj, float(prob), d))
                if cache is not None:
                    cache_pair(f, t_src, t_tgt, feat_p, feat_s, logits, s0, g0, ext=not cfg.cache_lite)
            del feat_p, feat_s, det

    if cache is not None and base_pairs and (cfg.edge_temp != 1.0 or cfg.edge_temp_auto > 0):
        # calibration of the linker's confidence: rebuild the ILP candidates from the gated logits at the chosen
        # temperature (per-video rule with edge_temp_auto); the exact inferred candidates are kept when T stays 1
        bp0 = np.concatenate(base_pairs)
        bl0 = np.concatenate(base_lse)
        lse0 = np.full(len(coords_so_far), np.nan)
        lse0[bl0[:, 0].astype(np.int64)] = bl0[:, 1]
        src_i, tgt_i = bp0[:, 0].astype(np.int64), bp0[:, 1].astype(np.int64)
        p_t, t_used = temper_edge_probs(src_i, tgt_i, bp0[:, 2], lse0, cfg.edge_temp, cfg.edge_temp_auto, cfg.edge_temp_min)
        if t_used != 1.0:
            keep = p_t > cfg.edge_threshold
            edges = sorted(
                ((int(i), int(j), float(p), float(np.linalg.norm(coords_so_far[i, 1:].astype(np.float32)
                                                                  - coords_so_far[j, 1:].astype(np.float32))))
                 for i, j, p in zip(src_i[keep], tgt_i[keep], p_t[keep])), key=lambda e: (e[0], -e[2]))
        print(f"  edge temperature {t_used:.3f}: {len(edges)} candidates", flush=True)

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr

    if cache is not None:
        n_s = len(coords)
        weak = (np.concatenate(weak_lists) if weak_lists else np.empty((0, 4), dtype=np.int16)).astype(np.float32)
        weak[:, 1:] *= ds_arr
        n_all = n_s + len(weak)

        def remap(code: np.ndarray) -> np.ndarray:
            code = code.astype(np.int64)
            return np.where(code >= 0, code, n_s + (-code - 1))

        ext = np.concatenate(ext_pairs) if ext_pairs else np.empty((0, 3))
        lse = np.concatenate(ext_lse) if ext_lse else np.empty((0, 3))
        lse_all = np.full(n_all, np.nan, dtype=np.float32)
        lse_strong = np.full(n_all, np.nan, dtype=np.float32)
        if len(lse):
            idx = remap(lse[:, 0])
            lse_all[idx], lse_strong[idx] = lse[:, 1], lse[:, 2]
        bp = np.concatenate(base_pairs) if base_pairs else np.empty((0, 3))
        bl = np.concatenate(base_lse) if base_lse else np.empty((0, 2))
        base_lse_arr = np.full(n_s, np.nan, dtype=np.float32)
        if len(bl):
            base_lse_arr[bl[:, 0].astype(np.int64)] = bl[:, 1]
        cand = np.array([(i, j, p) for i, j, p, _ in edges], dtype=np.float64).reshape(-1, 3)  # exact ILP input
        C = next((v.shape[1] for v in feat_store.values()), 0)
        feats = np.zeros((n_all, C), dtype=np.float16)
        for t, F in feat_store.items():
            s0, s1 = offsets[t]
            w0, w1 = weak_offsets.get(t, (0, 0))
            feats[s0:s1] = F[:s1 - s0]
            feats[n_s + w0:n_s + w1] = F[s1 - s0:s1 - s0 + (w1 - w0)]
        cache.update({
            "node_txyz": np.concatenate([coords, weak]).astype(np.int16),
            "node_prob": np.concatenate([np.concatenate(strong_prob_lists) if strong_prob_lists else np.empty(0, np.float32),
                                         np.concatenate(weak_prob_lists) if weak_prob_lists else np.empty(0, np.float32)]).astype(np.float16),
            "n_strong": np.int64(n_s), "feat": feats,
            "ext_src": remap(ext[:, 0]).astype(np.int32), "ext_tgt": remap(ext[:, 1]).astype(np.int32),
            "ext_logit": ext[:, 2].astype(np.float32), "lse_all": lse_all, "lse_strong": lse_strong,
            "base_src": bp[:, 0].astype(np.int32), "base_tgt": bp[:, 1].astype(np.int32),
            "base_logit": bp[:, 2].astype(np.float32), "base_lse": base_lse_arr,
            "cand_src": cand[:, 0].astype(np.int32), "cand_tgt": cand[:, 1].astype(np.int32),
            "cand_prob": cand[:, 2].astype(np.float32),
            "frozen": frozen, "T": np.int64(T), "voxel_um": np.array(ds.scale, dtype=np.float32),
            "downsample": np.array(downsample, dtype=np.int16), "det_threshold": np.float32(cfg.det_threshold),
            "cache_low_thr": np.float32(cfg.cache_low_thr), "cache_gate_um": np.float32(cfg.cache_gate_um),
        })
    if n_phase_added:
        print(f"  phase-shift detection: +{n_phase_added} peaks ({n_phase_added / max(len(coords), 1):.1%} of nodes)", flush=True)
    return coords.astype(np.int16), edges, {"retention_guard_fallback_frames": guard_fallbacks, "phase_peaks": n_phase_added}


def edge_uncertainty(raw_edges: list, prob_threshold: float) -> float:
    """Share of the selected (ILP) links whose probability is below the threshold."""
    probs = [float(e["edge_prob"]) for e in raw_edges if e.get("edge_prob") is not None]
    return float(np.mean(np.array(probs) < prob_threshold)) if probs else 0.0


def adaptive_ilp_cost(raw_edges: list, cfg: "Config") -> tuple[float | None, float]:
    """Disappearance cost to re-solve this video with (None = keep the first solution; < 0 = no ILP), and the
    linker uncertainty it was chosen from. The table maps uncertainty thresholds to costs; the highest threshold
    not above the video's uncertainty wins."""
    table = [list(x) for x in (cfg.ilp_adaptive_table or [])]
    if cfg.ilp_adaptive_disappearance > 0:
        table.append([cfg.ilp_adaptive_uncertainty, cfg.ilp_adaptive_disappearance])
    if not table:
        return None, float("nan")
    unc = edge_uncertainty(raw_edges, cfg.ilp_uncertain_prob)
    chosen = None
    for thr, cost in sorted(table):
        if unc >= thr:
            chosen = float(cost)
    if chosen is not None and chosen == cfg.ilp_disappearance:
        chosen = None
    return chosen, unc


def adaptive_ilp_needed(raw_edges: list, cfg: "Config") -> tuple[bool, float]:
    cost, unc = adaptive_ilp_cost(raw_edges, cfg)
    return cost is not None, unc


def no_ilp_graph(coords: np.ndarray, edges: list) -> tuple[dict, list]:
    """The graph repair would get with the ILP switched off: every detection and every candidate link."""
    nodes = {int(i): {"node_id": int(i), "t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3])}
             for i, c in enumerate(coords)}
    raw = [{"source_id": int(e[0]), "target_id": int(e[1]), "edge_prob": float(e[2])} for e in edges]
    return nodes, raw


def local_track_end_costs(n_nodes: int, src, tgt, prob, cfg: "Config") -> np.ndarray:
    """Per-node disappearance cost from the confidence of the node's best outgoing candidate link."""
    conf = np.zeros(n_nodes)
    if len(src):
        np.maximum.at(conf, np.asarray(src, dtype=np.int64), np.asarray(prob, dtype=np.float64))
    return cfg.ilp_local_dmin + (cfg.ilp_disappearance - cfg.ilp_local_dmin) * conf ** cfg.ilp_local_gamma


def build_graph_with_attrs(coords, edges: list, node_attrs: dict):
    """Same graph as the support pack's build_graph, plus float node attributes (e.g. per-node costs)."""
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x", *node_attrs]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0 if key in ("z", "y", "x") else 0.0)
    node_ids = graph.bulk_add_nodes([
        {"t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3]),
         **{k: float(v[i]) for k, v in node_attrs.items()}} for i, c in enumerate(coords)])
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([{"source_id": node_ids[int(e[0])], "target_id": node_ids[int(e[1])],
                               "edge_prob": float(e[2]), "edge_dist": float(e[3]) if len(e) > 3 else 0.0} for e in edges])
    return graph


def flow_penalized_probs(cache: "RichCache", cfg: "Config", raw_nodes: dict, raw_edges: list, use_weak: bool = False):
    """Candidate probabilities after a motion prior: logit -= penalty * (|d - u| / sigma)^2 / 2, re-normalised over
    the gated parents of each target. The flow u comes from the confident links of the first (baseline) solution."""
    flow = Flow(raw_nodes, raw_edges, cfg)
    if use_weak:
        src, tgt, logit = cache.ext_src, cache.ext_tgt, cache.ext_logit
    else:
        src, tgt, logit = cache.base_src, cache.base_tgt, cache.base_logit
    src, tgt = np.asarray(src).astype(np.int64), np.asarray(tgt).astype(np.int64)
    disp = cache.um[tgt] - cache.um[src]
    pred = np.zeros_like(disp)
    usrc, inv = np.unique(src, return_inverse=True)          # the prediction depends on the source only
    upred = np.stack([flow.at(int(cache.t[i]), cache.um[i], own=cache.um[i]) for i in usrc]) if len(usrc) else np.zeros((0, 3))
    pred[:] = upred[inv]
    resid = np.linalg.norm(disp - pred, axis=1)
    adj = np.asarray(logit, dtype=np.float64) - cfg.ilp_flow_penalty * (resid / cfg.ilp_flow_sigma_um) ** 2 / 2.0
    m = np.full(len(cache.txyz), -np.inf)
    np.maximum.at(m, tgt, adj)
    ssum = np.zeros(len(cache.txyz))
    np.add.at(ssum, tgt, np.exp(adj - m[tgt]))
    prob = np.exp(adj - (m[tgt] + np.log(ssum[tgt])))
    return src, tgt, prob


def ncc_pairs(frames, txyz: np.ndarray, src: np.ndarray, tgt: np.ndarray, half=(4, 16, 16), device: str = "cpu",
              batch: int = 1024) -> np.ndarray:
    """Best normalised cross-correlation (over ±1 voxel shifts) between the raw crop around each source (frame t)
    and the crop around its target (frame t+1). Batched with torch per frame pair; nan where a crop leaves the volume."""
    import torch
    out = np.full(len(src), np.nan, dtype=np.float32)
    if frames is None or not len(src):
        return out
    hz, hy, hx = half
    t_src = txyz[src, 0].astype(np.int64)
    ar = [torch.arange(-h, h + 1, device=device) for h in half]
    arb = [torch.arange(-h - 1, h + 2, device=device) for h in half]
    for t in np.unique(t_src):
        sel = np.nonzero(t_src == t)[0]
        va = torch.as_tensor(frames.get(int(t)).astype(np.float32), device=device)
        vb = torch.as_tensor(frames.get(int(t) + 1).astype(np.float32), device=device)
        Z, Y, X = va.shape
        for s0 in range(0, len(sel), batch):
            idx = sel[s0:s0 + batch]
            ca = torch.as_tensor(np.rint(txyz[src[idx], 1:]).astype(np.int64), device=device)
            cb = torch.as_tensor(np.rint(txyz[tgt[idx], 1:]).astype(np.int64), device=device)
            ok = ((ca[:, 0] - hz >= 0) & (ca[:, 0] + hz < Z) & (ca[:, 1] - hy >= 0) & (ca[:, 1] + hy < Y) &
                  (ca[:, 2] - hx >= 0) & (ca[:, 2] + hx < X) & (cb[:, 0] - hz - 1 >= 0) & (cb[:, 0] + hz + 1 < Z) &
                  (cb[:, 1] - hy - 1 >= 0) & (cb[:, 1] + hy + 1 < Y) & (cb[:, 2] - hx - 1 >= 0) & (cb[:, 2] + hx + 1 < X))
            if not bool(ok.any()):
                continue
            ca, cb, idx_ok = ca[ok], cb[ok], idx[ok.cpu().numpy()]
            A = va[(ca[:, 0, None] + ar[0])[:, :, None, None], (ca[:, 1, None] + ar[1])[:, None, :, None],
                   (ca[:, 2, None] + ar[2])[:, None, None, :]]
            B = vb[(cb[:, 0, None] + arb[0])[:, :, None, None], (cb[:, 1, None] + arb[1])[:, None, :, None],
                   (cb[:, 2, None] + arb[2])[:, None, None, :]]
            A = A - A.mean(dim=(1, 2, 3), keepdim=True)
            A = A / A.flatten(1).norm(dim=1).clamp_min(1e-6)[:, None, None, None]
            best = torch.full((len(idx_ok),), -2.0, device=device)
            for dz in range(3):
                for dy in range(3):
                    for dx in range(3):
                        Bs = B[:, dz:dz + 2 * hz + 1, dy:dy + 2 * hy + 1, dx:dx + 2 * hx + 1]
                        Bs = Bs - Bs.mean(dim=(1, 2, 3), keepdim=True)
                        v = (A * Bs).flatten(1).sum(1) / Bs.flatten(1).norm(dim=1).clamp_min(1e-6)
                        best = torch.maximum(best, v)
            out[idx_ok] = best.cpu().numpy()
    return out


LINK_FEATURES = ["logit", "gap_t", "rank_t", "n_t", "p_t", "q_s", "gap_s", "rank_s", "n_s", "dist", "dz",
                 "flow_res", "flow_res_rank_s", "feat_cos", "ncc", "ncc_rel_s", "ncc_rel_t", "prob_src", "prob_tgt"]
_LINK_MODELS: dict = {}


def link_working_set(cache: "RichCache", min_p: float = 0.01):
    """Base candidate pairs worth re-scoring: target-normalised or source-normalised probability above min_p."""
    src, tgt = np.asarray(cache.base_src).astype(np.int64), np.asarray(cache.base_tgt).astype(np.int64)
    logit = np.asarray(cache.base_logit, dtype=np.float64)
    p_t = np.exp(logit - cache.base_lse[tgt])
    m = np.full(len(cache.txyz), -np.inf)
    np.maximum.at(m, src, logit)
    ss = np.zeros(len(cache.txyz))
    np.add.at(ss, src, np.exp(logit - m[src]))
    q_s = np.exp(logit - m[src]) / ss[src]
    keep = (p_t > min_p) | (q_s > min_p)
    return np.nonzero(keep)[0]


def _group_stats(key: np.ndarray, val: np.ndarray, n: int, higher_better: bool = True):
    """Per-group max, rank (0 = best) and group size of val over the groups given by key."""
    v = val if higher_better else -val
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, key, np.where(np.isfinite(v), v, -np.inf))
    order = np.lexsort((-v, key))
    rank = np.empty(len(key), dtype=np.float64)
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    pos = np.arange(len(key)) - np.repeat(start, np.diff(np.r_[start, len(key)]))
    rank[order] = pos
    cnt = np.bincount(key, minlength=n)
    return mx, rank, cnt


def link_features(cache: "RichCache", k: np.ndarray, flow: "Flow", frames, device: str = "cpu") -> np.ndarray:
    """LINK_FEATURES for base pairs k (indices into cache.base_*)."""
    n = len(cache.txyz)
    src, tgt = np.asarray(cache.base_src)[k].astype(np.int64), np.asarray(cache.base_tgt)[k].astype(np.int64)
    logit = np.asarray(cache.base_logit, dtype=np.float64)[k]
    p_t = np.exp(logit - cache.base_lse[tgt])
    mt, rank_t, n_t = _group_stats(tgt, logit, n)
    ms, rank_s, n_s = _group_stats(src, logit, n)
    ss = np.zeros(n)
    np.add.at(ss, src, np.exp(logit - ms[src]))
    q_s = np.exp(logit - ms[src]) / ss[src]
    disp = cache.um[tgt] - cache.um[src]
    usrc, inv = np.unique(src, return_inverse=True)
    upred = np.stack([flow.at(int(cache.t[i]), cache.um[i], own=cache.um[i]) for i in usrc]) if len(usrc) else np.zeros((0, 3))
    resid = np.linalg.norm(disp - upred[inv], axis=1)
    _, fr_rank, _ = _group_stats(src, resid, n, higher_better=False)
    if cache.feat is not None:
        fa, fb = cache.feat[src].astype(np.float32), cache.feat[tgt].astype(np.float32)
        fcos = (fa * fb).sum(1) / np.maximum(np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1), 1e-9)
    else:
        fcos = np.full(len(k), np.nan)
    ncc = ncc_pairs(frames, cache.txyz, src, tgt, device=device)
    nc = np.where(np.isfinite(ncc), ncc, -np.inf)
    mxs = np.full(n, -np.inf)
    np.maximum.at(mxs, src, nc)
    mxt = np.full(n, -np.inf)
    np.maximum.at(mxt, tgt, nc)
    rel_s = np.where(np.isfinite(ncc) & np.isfinite(mxs[src]), mxs[src] - ncc, np.nan)
    rel_t = np.where(np.isfinite(ncc) & np.isfinite(mxt[tgt]), mxt[tgt] - ncc, np.nan)
    return np.column_stack([logit, mt[tgt] - logit, rank_t, n_t[tgt], p_t, q_s, ms[src] - logit, rank_s, n_s[src],
                            np.linalg.norm(disp, axis=1), np.abs(disp[:, 0]), resid, fr_rank, fcos, ncc, rel_s, rel_t,
                            cache.prob[src], cache.prob[tgt]]).astype(np.float64)


def rescored_probs(cache: "RichCache", cfg: "Config", flow: "Flow", frames, model_path: str, device: str = "cpu",
                   feature_file: str | None = None, drop: tuple = ()):
    """(src, tgt, prob) over the working set with the learned link model; prob renormalised over each target's parents.
    rescore_mode "tiebreak": no model; the transformer's probabilities stay, except for targets whose two best parents
    are nearly tied, where raw-crop cross-correlation re-weights them (mass per target preserved)."""
    k = X = None
    if cfg.rescore_mode == "rankbonus":                   # no features: reward mutual-best pairs before the per-child softmax
        k = link_working_set(cache)
        src, tgt = np.asarray(cache.base_src)[k].astype(np.int64), np.asarray(cache.base_tgt)[k].astype(np.int64)
        z = np.asarray(cache.base_logit, dtype=np.float64)[k]
        best_t = np.full(len(cache.txyz), -np.inf)
        np.maximum.at(best_t, tgt, z)
        best_s = np.full(len(cache.txyz), -np.inf)
        np.maximum.at(best_s, src, z)
        col, row = z >= best_t[tgt] - 1e-9, z >= best_s[src] - 1e-9
        z = z + cfg.rank_beta * (col + 0.5 * row + 0.5 * (col & row))
        m = np.full(len(cache.txyz), -np.inf)
        np.maximum.at(m, tgt, z)
        ss = np.zeros(len(cache.txyz))
        np.add.at(ss, tgt, np.exp(z - m[tgt]))
        return src, tgt, np.exp(z - m[tgt]) / ss[tgt]
    if feature_file and Path(feature_file).exists():     # lab: features precomputed on the GPU box
        with np.load(feature_file) as d:
            k, X = d["k"], d["X"]
    if k is None:
        k = link_working_set(cache)
        X = link_features(cache, k, flow, frames, device)
    src, tgt = np.asarray(cache.base_src)[k].astype(np.int64), np.asarray(cache.base_tgt)[k].astype(np.int64)
    if cfg.rescore_mode == "tiebreak":
        p = np.exp(np.asarray(cache.base_logit, dtype=np.float64)[k] - cache.base_lse[tgt])
        ncc = X[:, LINK_FEATURES.index("ncc")]
        order = np.lexsort((-p, tgt))
        ts, ps = tgt[order], p[order]
        first = np.r_[True, ts[1:] != ts[:-1]]
        second = np.r_[False, first[:-1]] & ~first
        p1 = dict(zip(ts[first].tolist(), ps[first].tolist()))
        p2 = dict(zip(ts[second].tolist(), ps[second].tolist()))
        tie = np.array([p2.get(g, 0.0) >= cfg.tiebreak_min_p2 and p1[g] - p2.get(g, 0.0) < cfg.tiebreak_gap
                        for g in tgt.tolist()])
        nc = np.where(np.isfinite(ncc), ncc, -np.inf)
        mx = np.full(len(cache.txyz), -np.inf)
        np.maximum.at(mx, tgt, nc)
        w = np.where(tie & np.isfinite(ncc) & np.isfinite(mx[tgt]), np.exp(cfg.tiebreak_beta * (nc - mx[tgt])), 1.0)
        pw = p * w
        tot, totw = np.zeros(len(cache.txyz)), np.zeros(len(cache.txyz))
        np.add.at(tot, tgt, p)
        np.add.at(totw, tgt, pw)
        return src, tgt, np.where(tie, pw / np.maximum(totw[tgt], 1e-12) * tot[tgt], p)
    import lightgbm as lgb
    if model_path not in _LINK_MODELS:
        _LINK_MODELS[model_path] = lgb.Booster(model_file=model_path)
    if drop:
        X = X[:, [i for i, f in enumerate(LINK_FEATURES) if f not in drop]]
    z = _LINK_MODELS[model_path].predict(X, raw_score=True)
    if cfg.rescore_blend < 1.0:     # keep part of the transformer's own logit
        z = cfg.rescore_blend * z + (1 - cfg.rescore_blend) * np.asarray(cache.base_logit, dtype=np.float64)[k]
    z = z / max(cfg.rescore_temp, 1e-6)
    m = np.full(len(cache.txyz), -np.inf)
    np.maximum.at(m, tgt, z)
    ss = np.zeros(len(cache.txyz))
    np.add.at(ss, tgt, np.exp(z - m[tgt]))
    return src, tgt, np.exp(z - m[tgt]) / ss[tgt]


def candidate_flow_graph(coords: np.ndarray, cand: list, cfg: "Config") -> tuple[dict, list]:
    """A first-pass graph for the flow without an ILP: confident candidate links that are mutual best matches."""
    best_parent: dict[int, tuple[float, int]] = {}
    best_child: dict[int, tuple[float, int]] = {}
    for e in cand:
        i, j, p = int(e[0]), int(e[1]), float(e[2])
        if p > best_parent.get(j, (-1.0, -1))[0]:
            best_parent[j] = (p, i)
        if p > best_child.get(i, (-1.0, -1))[0]:
            best_child[i] = (p, j)
    edges = [{"source_id": i, "target_id": j, "edge_prob": p} for j, (p, i) in best_parent.items()
             if p >= cfg.flow_conf_prob and best_child.get(i, (0.0, -1))[1] == j]
    used = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
    nodes = {i: {"node_id": i, "t": int(coords[i][0]), "z": float(coords[i][1]), "y": float(coords[i][2]),
                 "x": float(coords[i][3])} for i in used}
    return nodes, edges


def flow_penalized_candidates(cache: "RichCache", cfg: "Config", raw_nodes: dict, raw_edges: list) -> list:
    """ILP input rebuilt with the motion prior; same thresholds as the out-of-fold lab (edge_threshold, edge_max_um)."""
    src, tgt, prob = flow_penalized_probs(cache, cfg, raw_nodes, raw_edges)
    keep = prob > cfg.edge_threshold
    src, tgt, prob = src[keep], tgt[keep], prob[keep]
    dist = np.linalg.norm(cache.um[src] - cache.um[tgt], axis=1)
    if cfg.edge_max_um > 0:
        ok = dist <= cfg.edge_max_um
        src, tgt, prob, dist = src[ok], tgt[ok], prob[ok], dist[ok]
    return [(int(a), int(b), float(p), float(d)) for a, b, p, d in zip(src, tgt, prob, dist)]


def calibrate_mit(mit: np.ndarray, n_strong: int, mode: str, ref_json: str) -> np.ndarray:
    """Per-video calibration of the cell-state scores against an embryo shift: the video's logit quantiles (strong
    rows) of P(parent) and P(daughter) are mapped piecewise-linearly onto reference quantiles (mit_calib_ref), so a
    model that is systematically under-confident on an unseen embryo gets the reference tail back.  Below the median
    the logits are only shifted; above the last knot the last segment is extrapolated."""
    import json
    ref = json.loads(ref_json)
    levels = np.asarray(ref["levels"], dtype=np.float64)
    out = np.asarray(mit, dtype=np.float32).copy()
    rows = np.nonzero(np.isfinite(out[:n_strong, 1]))[0]
    if len(rows) < 200 or mode != "quantile":
        return out
    for col in (1, 2):
        p = np.clip(out[rows, col].astype(np.float64), 1e-4, 1 - 1e-4)
        lg = np.log(p / (1 - p))
        vq = np.percentile(lg, levels)
        rq = np.asarray(ref[str(col)], dtype=np.float64)
        keep = np.r_[True, np.diff(vq) > 1e-6]
        vq, rq = vq[keep], rq[keep]
        if len(vq) < 2:
            continue
        mapped = np.interp(lg, vq, rq)
        lo = lg < vq[0]
        mapped[lo] = lg[lo] + (rq[0] - vq[0])
        hi = lg > vq[-1]
        slope = (rq[-1] - rq[-2]) / max(vq[-1] - vq[-2], 1e-6)
        mapped[hi] = rq[-1] + (lg[hi] - vq[-1]) * slope
        out[rows, col] = 1.0 / (1.0 + np.exp(-mapped))
    out[rows, 0] = np.clip(1.0 - out[rows, 1] - out[rows, 2], 0.0, 1.0)
    return out


def cnn_node_costs(mit: np.ndarray, cfg: "Config", top2: np.ndarray | None = None) -> dict:
    """Per-node ILP costs from cell-state probabilities (normal, parent, daughter); nan scores leave the defaults.
    `top2` (optional): the linker's second-highest candidate probability per node, see second_best_child_prob."""
    mp = np.nan_to_num(mit[:, 1].astype(np.float64), nan=0.0)
    md = np.nan_to_num(mit[:, 2].astype(np.float64), nan=0.0)
    div_cost = cfg.ilp_division - cfg.ilp_div_gamma * mp
    if cfg.ilp_div_top2_beta > 0 and top2 is not None:      # the linker's second-best child pays part of the division
        div_cost = div_cost - cfg.ilp_div_top2_beta * np.maximum(np.asarray(top2, dtype=np.float64)[:len(mp)]
                                                                   - cfg.ilp_div_top2_tau, 0.0)
    return {"div_cost": np.maximum(div_cost, 0.0),
            "app_cost": cfg.ilp_appearance + cfg.ilp_app_gamma * md}


def second_best_child_prob(src: np.ndarray, prob: np.ndarray, n_nodes: int) -> np.ndarray:
    """Per source node: the second-highest candidate-edge probability leaving it (0 when fewer than two candidates)."""
    top2 = np.zeros(n_nodes, dtype=np.float64)
    if len(src) == 0:
        return top2
    src = np.asarray(src, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    order = np.lexsort((-prob, src))
    s, p = src[order], prob[order]
    first = np.r_[True, s[1:] != s[:-1]]
    second = np.r_[False, (~first[1:]) & first[:-1]]        # the element right after a group start
    top2[s[second]] = p[second]
    return top2


def solve_ilp(coords: np.ndarray, edges: list, models: TrackingModels, cfg: Config, mit: np.ndarray | None = None,
              node_cost: np.ndarray | None = None):
    """Global linking with the tracksdata ILP; returns node dicts and edge dicts of the solution.
    `node_cost` (optional, per detection): a cost every selected node pays (ilp_dc_weight: DeepCenter doubt)."""
    import tracksdata as td
    sp = models.support
    disappearance = cfg.ilp_disappearance
    division, appearance = cfg.ilp_division, cfg.ilp_appearance
    attrs = {}
    if cfg.ilp_local_dmin >= 0:
        attrs["d_cost"] = local_track_end_costs(len(coords), [e[0] for e in edges], [e[1] for e in edges],
                                                [e[2] for e in edges], cfg)
        disappearance = td.NodeAttr("d_cost")
    if cfg.ilp_div_cnn and mit is not None and len(mit) == len(coords):
        top2 = (second_best_child_prob(np.array([e[0] for e in edges]), np.array([e[2] for e in edges]), len(coords))
                if cfg.ilp_div_top2_beta > 0 else None)
        attrs.update(cnn_node_costs(mit, cfg, top2))
        division, appearance = td.NodeAttr("div_cost"), td.NodeAttr("app_cost")
        if cfg.ilp_div_edge_beta > 0 or cfg.ilp_div_edge_penalty > 0:   # mother -> daughter bonus / continuation penalty
            mp = np.nan_to_num(mit[:, 1].astype(np.float64), nan=0.0)
            md = np.nan_to_num(mit[:, 2].astype(np.float64), nan=0.0)
            edges = [(e[0], e[1], min(1.0, max(0.0, float(e[2]) + cfg.ilp_div_edge_beta * mp[e[0]] * md[e[1]]
                                                 - cfg.ilp_div_edge_penalty * md[e[1]] * (1.0 - mp[e[0]]))), *e[3:])
                     for e in edges]
    if node_cost is not None and len(node_cost) == len(coords):
        attrs["dc_cost"] = np.asarray(node_cost, dtype=np.float64)
    graph = build_graph_with_attrs(coords, edges, attrs) if attrs else sp.build_graph(coords, edges)
    if graph.num_edges() > 0:
        kw = {"node_weight": td.NodeAttr("dc_cost")} if "dc_cost" in attrs else {}
        solver = td.solvers.ILPSolver(
            edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=appearance,
            disappearance_weight=disappearance,
            division_weight=division,
            **kw,
        )
        with sp.suppress_output():
            graph = solver.solve(graph)
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes[nid] = {"node_id": nid, "t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]),
                      "x": float(row["x"])}
    raw_edges = []
    if graph.num_edges() > 0:
        for row in graph.edge_attrs().iter_rows(named=True):
            prob = row.get("edge_prob")
            raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                              "edge_prob": None if prob is None else float(prob)})
    return nodes, raw_edges


# =============================================================================================
# DeepCenter center prior (used only to confirm synthetic gap nodes and safe-division daughters)
# =============================================================================================

def _deepcenter_classes():
    import torch

    class ConvBlock(torch.nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            g = min(8, cout)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, 3, padding=1, bias=False), torch.nn.GroupNorm(g, cout),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(cout, cout, 3, padding=1, bias=False), torch.nn.GroupNorm(g, cout),
                torch.nn.SiLU(inplace=True))

        def forward(self, x):
            return self.block(x)

    class DeepCenterUNet3D(torch.nn.Module):
        def __init__(self, base_channels: int = 24):
            super().__init__()
            c = int(base_channels)
            self.enc1, self.down1 = ConvBlock(1, c), torch.nn.MaxPool3d(2, 2)
            self.enc2, self.down2 = ConvBlock(c, 2 * c), torch.nn.MaxPool3d(2, 2)
            self.enc3, self.down3 = ConvBlock(2 * c, 4 * c), torch.nn.MaxPool3d(2, 2)
            self.bottleneck = ConvBlock(4 * c, 8 * c)
            self.up3, self.dec3 = torch.nn.ConvTranspose3d(8 * c, 4 * c, 2, 2), ConvBlock(8 * c, 4 * c)
            self.up2, self.dec2 = torch.nn.ConvTranspose3d(4 * c, 2 * c, 2, 2), ConvBlock(4 * c, 2 * c)
            self.up1, self.dec1 = torch.nn.ConvTranspose3d(2 * c, c, 2, 2), ConvBlock(2 * c, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return DeepCenterUNet3D


class DeepCenter:
    def __init__(self, cfg: Config, device):
        import torch
        ckpt = torch.load(cfg.deepcenter_ckpt, map_location=device, weights_only=False)
        self.ccfg = SimpleNamespace(**ckpt.get("config", {}))
        self.model = _deepcenter_classes()(int(getattr(self.ccfg, "base_channels", 24)))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()
        self.device = device
        self.views = TTA_VIEWS[cfg.deepcenter_views]
        self.pool = int(getattr(self.ccfg, "pool_factor", 4))
        self.win_z, self.win_yx = cfg.deepcenter_win_z, cfg.deepcenter_win_yx
        self._cache: dict[int, np.ndarray] = {}

    def reset(self):
        self._cache.clear()

    def _heatmap(self, frames: FrameCache, t: int) -> np.ndarray:
        import torch
        if t in self._cache:
            return self._cache[t]
        vol = frames.get(t).astype(np.float32)
        z, y, x = vol.shape
        f = self.pool
        pooled = vol[:, :(y // f) * f, :(x // f) * f].reshape(z, y // f, f, x // f, f).mean(axis=(2, 4))
        lo = float(np.percentile(pooled, float(getattr(self.ccfg, "norm_lo_pct", 50.0))))
        hi = float(np.percentile(pooled, float(getattr(self.ccfg, "norm_hi_pct", 99.5))))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            image = np.zeros_like(pooled, dtype=np.float32)
        else:
            image = np.clip((pooled - lo) / (hi - lo), float(getattr(self.ccfg, "norm_clip_lo", -0.5)),
                            float(getattr(self.ccfg, "norm_clip_hi", 6.0))).astype(np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(image[None, None]).to(self.device)
            square = tensor.shape[-1] == tensor.shape[-2]
            views = [v for v in self.views if square or v not in ("r90", "r270", "t", "at")]
            acc = None
            for v in views:
                out = view_inv(self.model(view_fwd(tensor, v)), v)
                acc = out if acc is None else acc + out
            heat = torch.sigmoid(acc / len(views))[0, 0].cpu().numpy().astype(np.float32)
        self._cache[t] = heat
        while len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        return heat

    def score(self, frames: FrameCache, t: int, point: tuple) -> float | None:
        heat = self._heatmap(frames, int(t))
        z = int(round(point[0]))
        y = int(round(point[1] / self.pool))
        x = int(round(point[2] / self.pool))
        patch = heat[max(0, z - self.win_z):z + self.win_z + 1, max(0, y - self.win_yx):y + self.win_yx + 1,
                     max(0, x - self.win_yx):x + self.win_yx + 1]
        if patch.size == 0:
            return None
        s = float(patch.max())
        return s if np.isfinite(s) else None

    def accept(self, frames: FrameCache, t: int, point: tuple, threshold: float) -> bool:
        s = self.score(frames, t, point)
        return True if s is None else s >= threshold


# =============================================================================================
# Rich cache (weak peaks, gated logits, features) and the neighbour-consensus flow
# =============================================================================================

class RichCache:
    """The rich cache of one video (arrays produced by predict_video with `cache`); positions in µm."""

    def __init__(self, data: dict):
        self.txyz = np.asarray(data["node_txyz"]).astype(np.float64)          # original voxels
        self.prob = np.asarray(data["node_prob"]).astype(np.float64)
        self.n_strong = int(data["n_strong"])
        self.voxel_um = np.asarray(data["voxel_um"]).astype(np.float64)
        self.um = self.txyz[:, 1:] * self.voxel_um[None, :]
        self.t = self.txyz[:, 0].astype(np.int64)
        self.frozen = np.asarray(data["frozen"]).astype(bool)
        self.T = int(data["T"])
        self.feat = data.get("feat")
        self.ext_src, self.ext_tgt = data.get("ext_src"), data.get("ext_tgt")
        self.ext_logit = None if data.get("ext_logit") is None else np.asarray(data["ext_logit"]).astype(np.float64)
        self.lse_all = None if data.get("lse_all") is None else np.asarray(data["lse_all"]).astype(np.float64)
        self.lse_strong = None if data.get("lse_strong") is None else np.asarray(data["lse_strong"]).astype(np.float64)
        self.base_src, self.base_tgt = data.get("base_src"), data.get("base_tgt")
        self.base_logit = None if data.get("base_logit") is None else np.asarray(data["base_logit"]).astype(np.float64)
        self.base_lse = None if data.get("base_lse") is None else np.asarray(data["base_lse"]).astype(np.float64)
        self.cand_src, self.cand_tgt = data.get("cand_src"), data.get("cand_tgt")
        self.cand_prob = None if data.get("cand_prob") is None else np.asarray(data["cand_prob"]).astype(np.float64)
        self.dc_score = None if data.get("dc_score") is None else np.asarray(data["dc_score"]).astype(np.float64)
        self._trees: dict[tuple[int, str], tuple] = {}

    @classmethod
    def load(cls, path: Path) -> "RichCache":
        with np.load(path) as d:
            return cls({k: d[k] for k in d.files})

    def tree(self, t: int, kind: str = "all"):
        """(cKDTree over µm positions, node ids) of frame t; kind in {"all", "strong", "weak"}."""
        from scipy.spatial import cKDTree
        key = (int(t), kind)
        if key not in self._trees:
            idx = np.nonzero(self.t == int(t))[0]
            if kind == "strong":
                idx = idx[idx < self.n_strong]
            elif kind == "weak":
                idx = idx[idx >= self.n_strong]
            self._trees[key] = (cKDTree(self.um[idx]) if len(idx) else None, idx)
        return self._trees[key]

    def near(self, t: int, point_um: np.ndarray, radius: float, kind: str = "all") -> tuple[np.ndarray, np.ndarray]:
        """Node ids of frame t within `radius` µm of the point, sorted by distance, and their distances."""
        tree, idx = self.tree(t, kind)
        if tree is None:
            return np.empty(0, dtype=np.int64), np.empty(0)
        hits = tree.query_ball_point(point_um, radius)
        if not hits:
            return np.empty(0, dtype=np.int64), np.empty(0)
        hits = np.array(hits)
        d = np.linalg.norm(self.um[idx[hits]] - point_um[None, :], axis=1)
        order = np.argsort(d)
        return idx[hits[order]], d[order]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-axis weighted median of (n, 3) values."""
    out = np.zeros(values.shape[1])
    for a in range(values.shape[1]):
        order = np.argsort(values[:, a])
        cw = np.cumsum(weights[order])
        out[a] = values[order[np.searchsorted(cw, 0.5 * cw[-1])], a]
    return out


class Flow:
    """Neighbour-consensus motion field per frame transition, from confident links of the (ILP) graph.

    The displacement predicted for a cell is the Gaussian-weighted median of the displacements of confidently
    linked neighbours (its own link excluded), or the frame's global drift when there are too few of them.
    """

    def __init__(self, nodes: dict, edges: list, cfg: Config):
        from scipy.spatial import cKDTree
        self.cfg = cfg
        by_t: dict[int, list] = {}
        for e in edges:
            p = e.get("edge_prob")
            s, g = nodes.get(e["source_id"]), nodes.get(e["target_id"])
            if s is None or g is None or g["t"] != s["t"] + 1 or p is None or not np.isfinite(float(p)):
                continue
            if float(p) < cfg.flow_conf_prob:
                continue
            ps, pg = point_um(s), point_um(g)
            by_t.setdefault(int(s["t"]), []).append((ps, pg - ps))
        self.fields: dict[int, tuple] = {}
        for t, items in by_t.items():
            P = np.stack([a for a, _ in items])
            D = np.stack([b for _, b in items])
            self.fields[t] = (cKDTree(P), P, D, np.median(D, axis=0))
        self.n_links = sum(len(v) for v in by_t.values())

    def drift(self, t: int) -> np.ndarray:
        f = self.fields.get(int(t))
        return f[3] if f is not None else np.zeros(3)

    def at(self, t: int, p: np.ndarray, own: np.ndarray | None = None) -> np.ndarray:
        """Predicted displacement (µm) over the transition t -> t+1 at position p (µm)."""
        f = self.fields.get(int(t))
        if f is None:
            return np.zeros(3)
        tree, P, D, g = f
        idx = tree.query_ball_point(p, self.cfg.flow_radius_um)
        if own is not None and idx:
            idx = [i for i in idx if np.abs(P[i] - own).max() > 1e-6]
        if len(idx) < self.cfg.flow_min_neighbors:
            return g
        idx = np.array(idx)
        d = np.linalg.norm(P[idx] - p[None, :], axis=1)
        w = np.exp(-0.5 * (d / self.cfg.flow_sigma_um) ** 2) + 1e-9
        if self.cfg.flow_stat == "mean":
            return (D[idx] * w[:, None]).sum(0) / w.sum()
        return _weighted_median(D[idx], w)

    def at_many(self, t: int, P: np.ndarray) -> np.ndarray:
        """`at` for many positions (n, 3) at once: one tree query per frame instead of one per position.
        Rows whose neighbourhood may be truncated by the k limit fall back to `at`, so the result is identical."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        f = self.fields.get(int(t))
        if f is None or not len(P):
            return np.zeros_like(P)
        tree, Pf, D, g = f
        r, sig = self.cfg.flow_radius_um, self.cfg.flow_sigma_um
        K = min(64, len(Pf))
        d, idx = tree.query(P, k=K, distance_upper_bound=r)
        d, idx = d.reshape(len(P), -1), idx.reshape(len(P), -1)
        valid = np.isfinite(d)
        cnt = valid.sum(1)
        out = np.tile(g, (len(P), 1))
        full = (cnt >= K) & (K < len(Pf))
        rows = (cnt >= self.cfg.flow_min_neighbors) & ~full
        if self.cfg.flow_stat == "mean":
            w = np.where(valid, np.exp(-0.5 * (np.where(valid, d, 0.0) / sig) ** 2) + 1e-9, 0.0)
            num = (D[np.where(valid, idx, 0)] * w[:, :, None]).sum(1)
            out[rows] = num[rows] / w.sum(1)[rows, None]
        else:
            for i in np.nonzero(rows)[0]:
                v = valid[i]
                out[i] = _weighted_median(D[idx[i, v]], np.exp(-0.5 * (d[i, v] / sig) ** 2) + 1e-9)
        for i in np.nonzero(full)[0]:
            out[i] = self.at(t, P[i])
        return out

    def forward(self, t: int, p: np.ndarray, steps: int = 1) -> np.ndarray:
        """Position advected from frame t through `steps` transitions."""
        p = np.array(p, dtype=np.float64)
        for k in range(steps):
            p = p + self.at(t + k, p)
        return p

    def forward_many(self, t: int, P: np.ndarray, steps: int = 1) -> np.ndarray:
        """`forward` for many positions (n, 3) at once."""
        P = np.array(P, dtype=np.float64).reshape(-1, 3)
        for k in range(steps):
            P = P + self.at_many(t + k, P)
        return P

    def backward(self, t: int, p: np.ndarray, steps: int = 1) -> np.ndarray:
        """Position advected backwards from frame t through `steps` transitions (field of t-1-k evaluated at p)."""
        p = np.array(p, dtype=np.float64)
        for k in range(steps):
            p = p - self.at(t - 1 - k, p)
        return p


# =============================================================================================
# Learned divisions: candidate triplets and their features
# =============================================================================================

def temper_edge_probs(src: np.ndarray, tgt: np.ndarray, logit: np.ndarray, lse: np.ndarray, temp: float = 1.0,
                      auto: float = 0.0, t_min: float = 0.6) -> tuple[np.ndarray, float]:
    """Linker probabilities of the gated pairs at another softmax temperature: (prob, temperature used).

    lse[t] is the full log-sum-exp over all parents of target t (as inferred); the mass outside the gated pairs is kept
    as one pseudo-source so that temp = 1 reproduces the inferred probabilities exactly. With auto > 0 the temperature
    is chosen per video: an under-confident video (median top-1 probability below `auto`) is sharpened until it reaches
    `auto` (never below t_min); a confident video keeps temp = 1.
    """
    tgt = np.asarray(tgt, dtype=np.int64)
    logit = np.asarray(logit, dtype=np.float64)
    lse = np.asarray(lse, dtype=np.float64)
    mass = np.zeros(len(lse))
    np.add.at(mass, tgt, np.exp(logit - lse[tgt]))
    tail_logit = lse + np.log(np.clip(1.0 - mass, 1e-12, None))

    def probs_at(T):
        z, zt = logit / T, tail_logit / T
        m = zt.copy()
        np.maximum.at(m, tgt, z)
        ssum = np.exp(zt - m)
        np.add.at(ssum, tgt, np.exp(z - m[tgt]))
        return np.exp(z - m[tgt]) / ssum[tgt]

    def median_top1(T):
        top = np.full(len(lse), -1.0)
        np.maximum.at(top, tgt, probs_at(T))
        return float(np.median(top[top >= 0])) if (top >= 0).any() else 1.0

    if auto > 0:
        if median_top1(1.0) >= auto:
            temp = 1.0
        elif median_top1(t_min) < auto:
            temp = t_min
        else:
            lo, hi = t_min, 1.0
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if median_top1(mid) < auto:
                    hi = mid
                else:
                    lo = mid
            temp = lo
    return probs_at(temp), float(temp)


DIV_FEATURES = [
    "d_sc1", "d_sq", "d_c1q", "fd_sc1", "fd_sq", "angle", "mid_off", "sym", "dz_c1q",
    "q_has_parent", "d_xq", "x_nchild", "q_len", "c1_len", "s_back", "x_back",
    "p_sq", "logit_sq", "p_sc1", "logit_sc1", "q_best_p", "q_s_rank", "p_xq", "logit_xq",
    "cos_sq", "cos_sc1", "cos_c1q", "ncc_sis", "ncc_sq", "ncc_sc1",
    "prob_s", "prob_c1", "prob_q", "dc_s", "dc_c1", "dc_q",
    "div_gap", "dens_s", "n_cand_q", "trel",
    "I_s", "I_s_rel", "I_s_jump", "I_c1", "I_q", "I_sis_ratio", "I_q_vs_s", "I_c1_vs_s",
    "q_dropped", "P_s", "P_c1", "P_q",
    "mp_s", "md_s", "mp_c1", "md_c1", "mp_q", "md_q",
    "mp_x", "md_x",
]
_DIV_MODELS: dict = {}
MIT_HZ, MIT_HY = 4, 24     # cell-state crops: 9 z-slices x 48 x 48 px over frames t-1, t, t+1


def mitosis_build_model(in_ch: int = 3):
    """Small 3D CNN: (in_ch frames, 9, 48, 48) -> logits for normal / dividing parent / fresh daughter."""
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                             nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
    return nn.Sequential(block(in_ch, 24), nn.MaxPool3d((1, 2, 2)), block(24, 48), nn.MaxPool3d((1, 2, 2)),
                         block(48, 96), nn.MaxPool3d((3, 2, 2)), block(96, 128), nn.AdaptiveAvgPool3d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 3))


def cuda_device_for_worker():
    """Spread lab workers over the visible GPUs (one per process id); the pipeline process gets its own GPU anyway."""
    import torch
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{os.getpid() % torch.cuda.device_count()}")


class MitosisEnsemble:
    """Several cell-state checkpoints voting in probability space; forward returns log(mean prob), so softmax(...) = mean."""

    def __init__(self, models):
        self.models = models
        self.mit_frames = getattr(models[0], "mit_frames", 3)
        self.mit_hz = getattr(models[0], "mit_hz", MIT_HZ)

    def __call__(self, x):
        import torch
        p = sum(torch.softmax(m(x).float(), dim=1) for m in self.models) / len(self.models)
        return torch.log(p.clamp_min(1e-9))

    def eval(self):
        return self


def mitosis_load(path: str, device):
    """One checkpoint, or several separated by commas (averaged probabilities)."""
    import torch
    key = ("mitosis", path, str(device))
    if key not in _DIV_MODELS:
        models = []
        for one in [q.strip() for q in path.split(",") if q.strip()]:
            state = torch.load(one, map_location=device)
            hz = int(state.pop("_hz")) if "_hz" in state else MIT_HZ      # crop half depth in slices (4 = 9 slices)
            in_ch = int(state["0.0.weight"].shape[1])                # frames stacked as channels (3 or 5)
            model = mitosis_build_model(in_ch).to(device)
            model.load_state_dict(state)
            model.mit_frames = in_ch
            model.mit_hz = hz
            models.append(model.eval())
        _DIV_MODELS[key] = models[0] if len(models) == 1 else MitosisEnsemble(models)
    return _DIV_MODELS[key]


def mitosis_norm_crops(x):
    """Standardise every crop on its own (mean 0, std 1 over frames x z x y x x): embryo-independent contrast."""
    m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    sd = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-3
    return (x - m) / sd


def mitosis_score_nodes(model, zarr_path: Path, txyz, device, batch: int = 384, views: int = 4,
                        norm: str = "video") -> np.ndarray:
    """Cell-state probabilities (normal, parent, daughter) for nodes (t, z, y, x voxels); crops gathered on the device."""
    import torch
    hz, hy = int(getattr(model, "mit_hz", MIT_HZ)), MIT_HY
    txyz = np.asarray(txyz, dtype=np.float64).reshape(-1, 4)
    out = np.full((len(txyz), 3), np.nan, dtype=np.float32)
    if not len(txyz):
        return out
    T = zarr_meta(zarr_path)[0][0]
    f0 = read_frame(zarr_path, 0)
    lo, hi = float(np.percentile(f0, 1.0)), float(np.percentile(f0, 99.8))
    ar_z = torch.arange(2 * hz + 1, device=device)
    ar_y = torch.arange(2 * hy, device=device)
    frames: dict = {}

    def frame(t):
        t = min(max(int(t), 0), T - 1)
        if t not in frames:
            v = torch.as_tensor(read_frame(zarr_path, t).astype(np.float32), device=device)
            v = torch.clamp((v - lo) / max(hi - lo, 1e-6), 0, 1.5)
            frames[t] = torch.nn.functional.pad(v, (hy, hy, hy, hy, hz, hz))
        return frames[t]
    ts = txyz[:, 0].astype(np.int64)
    half = int(getattr(model, "mit_frames", 3)) // 2
    with torch.no_grad():
        for t in np.unique(ts):
            for old in [k for k in frames if k < t - half]:
                frames.pop(old)
            F = torch.stack([frame(t + d) for d in range(-half, half + 1)])
            sel = np.nonzero(ts == t)[0]
            for s0 in range(0, len(sel), batch):
                ids = sel[s0:s0 + batch]
                c = torch.as_tensor(np.rint(txyz[ids, 1:]).astype(np.int64), device=device)
                c[:, 0] = c[:, 0].clamp(0, F.shape[1] - 2 * hz - 1)
                c[:, 1] = c[:, 1].clamp(0, F.shape[2] - 2 * hy - 1)
                c[:, 2] = c[:, 2].clamp(0, F.shape[3] - 2 * hy - 1)
                x = F[:, (c[:, 0, None] + ar_z)[:, :, None, None], (c[:, 1, None] + ar_y)[:, None, :, None],
                      (c[:, 2, None] + ar_y)[:, None, None, :]].permute(1, 0, 2, 3, 4)
                if norm == "crop":
                    x = mitosis_norm_crops(x)
                vs = [x, torch.flip(x, dims=[4]), torch.flip(x, dims=[3]), torch.rot90(x, 1, dims=[3, 4])][:max(1, int(views))]
                prob = sum(torch.softmax(model(v).float(), dim=1) for v in vs) / len(vs)
                out[ids] = prob.cpu().numpy()
    return out


PAIR_HZ, PAIR_HY = 7, 32         # division-pair crops: 15 z-slices x 64 x 64 px over frames t, t+1, t+2 + 2 marker channels


def pair_marker(center_vox, points_vox, hz: int, hy: int) -> np.ndarray:
    """Gaussian blobs (sigma 1 slice, 2.5 px) at the given voxel points, drawn in a (2hz+1, 2hy, 2hy) crop whose
    origin is round(center) - (hz, hy, hy): the marker channels tell the pair model which triplet to judge."""
    c = np.rint(np.asarray(center_vox, dtype=np.float64))
    out = np.zeros((2 * hz + 1, 2 * hy, 2 * hy), dtype=np.float32)
    zz = np.arange(2 * hz + 1, dtype=np.float32)[:, None, None]
    yy = np.arange(2 * hy, dtype=np.float32)[None, :, None]
    xx = np.arange(2 * hy, dtype=np.float32)[None, None, :]
    for p in points_vox:
        pz, py, px = (np.asarray(p, dtype=np.float64) - (c - np.array([hz, hy, hy]))).astype(np.float32)
        out = np.maximum(out, np.exp(-0.5 * (((zz - pz) / 1.0) ** 2 + ((yy - py) / 2.5) ** 2 + ((xx - px) / 2.5) ** 2)))
    return out.astype(np.float32)


_PAIR_MODELS: dict = {}


def pair_load(path: str, device):
    """The division-pair checkpoint (state dict with `_hz`), cached per process."""
    import torch
    key = ("pair", path, str(device))
    if key not in _PAIR_MODELS:
        state = torch.load(path, map_location=device)
        hz = int(state.pop("_hz")) if "_hz" in state else PAIR_HZ
        model = mitosis_build_model(int(state["0.0.weight"].shape[1])).to(device)
        model.load_state_dict(state)
        model.mit_hz = hz
        _PAIR_MODELS[key] = model.eval()
    return _PAIR_MODELS[key]


def pair_score_triplets(model, zarr_path: Path, trips: list, device, batch: int = 24, views: int = 4) -> np.ndarray:
    """P(real division) for triplets (t, P_vox, D1_vox, D2_vox) with P at frame t and the daughters at t+1: crops of
    frames t, t+1, t+2 centred on the daughters' midpoint plus the two marker channels, flip TTA."""
    import torch
    hz, hy = int(getattr(model, "mit_hz", PAIR_HZ)), PAIR_HY
    out = np.full(len(trips), np.nan, dtype=np.float32)
    if not trips:
        return out
    T = zarr_meta(zarr_path)[0][0]
    f0 = read_frame(zarr_path, 0)
    lo, hi = float(np.percentile(f0, 1.0)), float(np.percentile(f0, 99.8))
    frames: dict = {}                                         # crops are cut on the CPU: the GPU only sees batches

    def frame(t):
        t = min(max(int(t), 0), T - 1)
        if t not in frames:
            frames[t] = np.clip((read_frame(zarr_path, t).astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1.5)
        return frames[t]

    def cut(vol, c):
        z0, y0, x0 = int(c[0]) - hz, int(c[1]) - hy, int(c[2]) - hy
        o = np.zeros((2 * hz + 1, 2 * hy, 2 * hy), dtype=np.float32)
        zs, ys, xs_ = max(0, z0), max(0, y0), max(0, x0)
        ze, ye, xe = min(vol.shape[0], z0 + o.shape[0]), min(vol.shape[1], y0 + o.shape[1]), min(vol.shape[2], x0 + o.shape[2])
        if ze > zs and ye > ys and xe > xs_:
            o[zs - z0:ze - z0, ys - y0:ye - y0, xs_ - x0:xe - x0] = vol[zs:ze, ys:ye, xs_:xe]
        return o
    order = sorted(range(len(trips)), key=lambda k: trips[k][0])
    with torch.no_grad():
        for s0 in range(0, len(order), batch):
            ids = order[s0:s0 + batch]
            ts = sorted({int(trips[k][0]) for k in ids})
            for old in [k for k in frames if k < ts[0]]:
                frames.pop(old)
            xs = np.zeros((len(ids), 5, 2 * hz + 1, 2 * hy, 2 * hy), dtype=np.float32)
            for j, k in enumerate(ids):
                t, P, D1, D2 = trips[k]
                c = np.rint((np.asarray(D1, dtype=np.float64) + np.asarray(D2, dtype=np.float64)) / 2.0).astype(np.int64)
                for ch, d in enumerate((0, 1, 2)):
                    xs[j, ch] = cut(frame(t + d), c)
                xs[j, 3] = pair_marker(c, [P], hz, hy)
                xs[j, 4] = pair_marker(c, [D1, D2], hz, hy)
            x = torch.as_tensor(xs, device=device)
            vs = [x, torch.flip(x, dims=[4]), torch.flip(x, dims=[3]), torch.rot90(x, 1, dims=[3, 4])][:max(1, int(views))]
            prob = 0
            for v in vs:                                       # one view at a time, bf16: 8 lab workers share a 16 GB GPU
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = model(v)
                prob = prob + torch.softmax(logits.float(), dim=1)
            prob = prob / len(vs)
            out[np.array(ids)] = prob[:, 1].cpu().numpy()
            del x, vs, logits, prob
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


CENTER_HZ, CENTER_HY = 3, 16     # click-offset crops: 7 z-slices x 32 x 32 px, single frame


def center_build_model(in_ch: int = 1, head: str = "avg"):
    """Small 3D CNN regressing the annotator click minus the detection (dz, dy, dx voxels) from a crop.

    in_ch = number of frames stacked as channels (1: frame t; 3: t-1, t, t+1); head "avg" pools the position away,
    head "flat" keeps the 7 x 8 x 8 feature positions (8 channels) before the linear layers.
    """
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
    body = [block(in_ch, 16), block(16, 32), nn.MaxPool3d((1, 2, 2)), block(32, 64), nn.MaxPool3d((1, 2, 2)), block(64, 96)]
    if head == "flat":
        n_pos = (2 * CENTER_HZ + 1) * (CENTER_HY // 2) ** 2
        top = [nn.Conv3d(96, 8, 1), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(8 * n_pos, 64), nn.ReLU(inplace=True),
               nn.Linear(64, 3)]
    else:
        top = [nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(96, 32), nn.ReLU(inplace=True), nn.Linear(32, 3)]
    return nn.Sequential(*body, *top)


def center_load(path: str, device):
    """A saved centre model: either {"state", "frames", "head"} or a bare state dict of the 1-frame/avg model."""
    import torch
    obj = torch.load(path, map_location=device)
    meta = {"frames": 1, "head": "avg"}
    state = obj
    if isinstance(obj, dict) and "state" in obj:
        meta.update({k: obj[k] for k in ("frames", "head") if k in obj})
        state = obj["state"]
    model = center_build_model(int(meta["frames"]), str(meta["head"])).to(device)
    model.load_state_dict(state)
    model.center_meta = meta
    return model.eval()


def center_score_nodes(model, zarr_path: Path, txyz, device, batch: int = 2048, tta: bool = False) -> np.ndarray:
    """Click-offset vectors (dz, dy, dx voxels) for nodes (t, z, y, x voxels); crops gathered on the device.

    With tta the prediction is averaged over the y/x flips (the flipped component changes sign).
    """
    import torch
    hz, hy = CENTER_HZ, CENTER_HY
    n_frames = int(getattr(model, "center_meta", {}).get("frames", 1))
    txyz = np.asarray(txyz, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((len(txyz), 3), dtype=np.float32)
    if not len(txyz):
        return out
    T = zarr_meta(zarr_path)[0][0]
    f0 = read_frame(zarr_path, 0)
    lo, hi = float(np.percentile(f0, 1.0)), float(np.percentile(f0, 99.8))
    ar_z = torch.arange(2 * hz + 1, device=device)
    ar_y = torch.arange(2 * hy, device=device)
    signs = [(None, (1.0, 1.0, 1.0))]
    if tta:
        signs += [([3], (1.0, -1.0, 1.0)), ([4], (1.0, 1.0, -1.0)), ([3, 4], (1.0, -1.0, -1.0))]
    frames: dict = {}

    def frame(t):
        t = min(max(int(t), 0), T - 1)
        if t not in frames:
            v = torch.as_tensor(read_frame(zarr_path, t).astype(np.float32), device=device)
            frames[t] = torch.nn.functional.pad(torch.clamp((v - lo) / max(hi - lo, 1e-6), 0, 1.5), (hy, hy, hy, hy, hz, hz))
        return frames[t]
    ts = txyz[:, 0].astype(np.int64)
    with torch.no_grad():
        for t in np.unique(ts):
            for old in [k for k in frames if k < t - 1]:
                frames.pop(old)
            F = torch.stack([frame(t - 1), frame(t), frame(t + 1)]) if n_frames == 3 else frame(t)[None]
            sel = np.nonzero(ts == t)[0]
            for s0 in range(0, len(sel), batch):
                ids = sel[s0:s0 + batch]
                c = torch.as_tensor(np.rint(txyz[ids, 1:]).astype(np.int64), device=device)
                c[:, 0] = c[:, 0].clamp(0, F.shape[1] - 2 * hz - 1)
                c[:, 1] = c[:, 1].clamp(0, F.shape[2] - 2 * hy - 1)
                c[:, 2] = c[:, 2].clamp(0, F.shape[3] - 2 * hy - 1)
                x = F[:, (c[:, 0, None] + ar_z)[:, :, None, None], (c[:, 1, None] + ar_y)[:, None, :, None],
                      (c[:, 2, None] + ar_y)[:, None, None, :]].permute(1, 0, 2, 3, 4)
                pred = 0
                for dims, sg in signs:
                    v = x if dims is None else torch.flip(x, dims=dims)
                    pred = pred + model(v).float() * torch.tensor(sg, device=device)
                out[ids] = (pred / len(signs)).cpu().numpy()
            del F
    return out


def division_model(path: str):
    if path not in _DIV_MODELS:
        import lightgbm as lgb
        _DIV_MODELS[path] = lgb.Booster(model_file=path)
    return _DIV_MODELS[path]


def _crop3(vol, c, half, shift=(0, 0, 0)):
    lo = [int(round(c[i])) + shift[i] - half[i] for i in range(3)]
    hi = [lo[i] + 2 * half[i] + 1 for i in range(3)]
    if min(lo) < 0 or any(hi[i] > vol.shape[i] for i in range(3)):
        return None
    return vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)


def _ncc_best(va, ca, vb, cb, half=(4, 16, 16)):
    """Best normalised cross-correlation of the crop around ca (in va) with crops around cb (in vb) within ±1 voxel."""
    A = _crop3(va, ca, half)
    if A is None:
        return float("nan")
    A = A - A.mean()
    na = float(np.sqrt((A * A).sum()))
    best = float("nan")
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                B = _crop3(vb, cb, half, (dz, dy, dx))
                if B is None:
                    continue
                B = B - B.mean()
                den = na * float(np.sqrt((B * B).sum()))
                if den > 0:
                    v = float((A * B).sum() / den)
                    if not best >= v:
                        best = v
    return best


def division_candidates(nodes: dict, edges: list, cache: "RichCache", flow: "Flow | None", frames, cfg: "Config",
                        dc_fn=None) -> dict:
    """Fork candidates (s, c1, q) and their DIV_FEATURES matrix.

    s: node at t with exactly one child c1 (at t+1); q: another node at t+1 within cfg.div_radius_um of s's
    flow-predicted position, not already s's child. q may have another parent x (a possible thief).
    """
    from scipy.spatial import cKDTree
    children: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    for e in edges:
        children.setdefault(e["source_id"], []).append(e["target_id"])
        parent[e["target_id"]] = e["source_id"]
    by_t: dict[int, list[int]] = {}
    for nid, n in nodes.items():
        by_t.setdefault(int(n["t"]), []).append(nid)
    pos = {nid: point_um(n) for nid, n in nodes.items()}

    def cidx(nid):
        n = nodes[nid]
        if nid < len(cache.t) and cache.t[nid] == n["t"] and np.abs(cache.um[nid] - pos[nid]).max() < 0.6:
            return nid
        ids, _ = cache.near(int(n["t"]), pos[nid], 0.6)
        return int(ids[0]) if len(ids) else None

    cmap = {nid: cidx(nid) for nid in nodes}
    pseudo: dict[int, dict] = {}              # dropped strong detections, keyed by -(cache index + 1)
    if cfg.div_use_dropped:
        used = set(v for v in cmap.values() if v is not None)
        for t_ in by_t:
            idx = np.nonzero((cache.t == t_) & (np.arange(len(cache.t)) < cache.n_strong))[0]
            if not len(idx):
                continue
            tr = cKDTree(np.stack([pos[i] for i in by_t[t_]]))
            for ci in idx:
                if int(ci) in used:
                    continue
                if tr.query(cache.um[ci])[0] < 2.0:      # already represented by a graph node
                    continue
                pid = -(int(ci) + 1)
                pseudo[pid] = {"node_id": pid, "t": int(t_), "z": float(cache.txyz[ci, 1]), "y": float(cache.txyz[ci, 2]),
                               "x": float(cache.txyz[ci, 3])}
                pos[pid] = cache.um[ci]
                cmap[pid] = int(ci)
    pseudo_by_t: dict[int, list[int]] = {}
    for pid, n in pseudo.items():
        pseudo_by_t.setdefault(n["t"], []).append(pid)
    pair = {}
    if cache.base_src is not None:
        pair = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(cache.base_src.tolist(), cache.base_tgt.tolist()))}
        tgt_order = np.argsort(cache.base_tgt, kind="stable")
        tgt_sorted = cache.base_tgt[tgt_order]

    def link(s, g):
        cs, cg = cmap.get(s), cmap.get(g)
        if cs is None or cg is None:
            return float("nan"), float("nan")
        k = pair.get((cs, cg))
        if k is None:
            return float("nan"), float("nan")
        lg = float(cache.base_logit[k])
        return float(np.exp(lg - cache.base_lse[cg])), lg

    po_memo: dict[int, tuple] = {}

    def parents_of(g):
        """(best parent probability, rank of each cached parent by logit) for cached node g."""
        if g in po_memo:
            return po_memo[g]
        cg = cmap.get(g)
        if cg is None or not len(pair):
            return float("nan"), {}
        lo, hi = np.searchsorted(tgt_sorted, cg), np.searchsorted(tgt_sorted, cg, side="right")
        ks = tgt_order[lo:hi]
        if not len(ks):
            po_memo[g] = (float("nan"), {})
            return po_memo[g]
        lg = cache.base_logit[ks]
        order = np.argsort(-lg)
        ranks = {int(cache.base_src[ks[o]]): r for r, o in enumerate(order)}
        po_memo[g] = (float(np.exp(lg.max() - cache.base_lse[cg])), ranks)
        return po_memo[g]

    rl_memo: dict[tuple, int] = {}

    def run_len(start, forward, cap=20):
        key = (start, forward)
        if key in rl_memo:
            return rl_memo[key]
        n, cur = 0, start
        while n < cap:
            nxt = (children.get(cur) if forward else ([parent[cur]] if cur in parent else None))
            if not nxt or len(nxt) != 1:
                break
            cur = nxt[0]
            n += 1
        rl_memo[key] = n
        return n

    fn_memo: dict[int, np.ndarray] = {}

    def funit(c):
        """Unit-norm transformer feature of cached detection c (memoised)."""
        if c not in fn_memo:
            f = cache.feat[c].astype(np.float64)
            fn_memo[c] = f / max(1e-9, math.sqrt(float(f @ f)))
        return fn_memo[c]

    def cos(a, b):
        ca, cb = cmap.get(a), cmap.get(b)
        if cache.feat is None or ca is None or cb is None:
            return float("nan")
        return float(funit(ca) @ funit(cb))

    def _n(v):
        return math.sqrt(float(v @ v))

    def cval(arr, nid):
        c = cmap.get(nid)
        return float(arr[c]) if (arr is not None and c is not None) else float("nan")

    mit = getattr(cache, "mit", None)          # cell-state scores (normal, parent, daughter) per cached detection

    def mitv(nid, k):
        c = cmap.get(nid)
        if mit is None or c is None or c >= len(mit):
            return float("nan")
        return float(mit[c, k])

    dc_memo: dict[int, float] = {}

    def dcv(nid):
        """DeepCenter score at a node: the live model in the pipeline (dc_fn), the cached score in the lab."""
        if nid not in dc_memo:
            v = None
            if dc_fn is not None:
                n = nodes[nid] if nid in nodes else pseudo[nid]
                try:
                    v = dc_fn(int(n["t"]), (float(n["z"]), float(n["y"]), float(n["x"])))
                except Exception:
                    v = None
            dc_memo[nid] = float(v) if v is not None else cval(cache.dc_score, nid)
        return dc_memo[nid]

    frame_norm: dict[int, float] = {}
    inten_cache: dict[int, tuple] = {}

    def inten(nid):
        """(core mean, context mean) of the raw frame around a node, divided by the frame's 99.5th percentile."""
        if nid in inten_cache:
            return inten_cache[nid]
        val = (float("nan"), float("nan"), float("nan"))
        if frames is not None and nid is not None:
            try:
                n = nodes[nid] if nid in nodes else pseudo[nid]
                t = int(n["t"])
                vol = frames.get(t)
                if t not in frame_norm:
                    frame_norm[t] = max(float(np.percentile(vol[::2, ::4, ::4], 99.5)), 1e-6)
                c = (float(n["z"]), float(n["y"]), float(n["x"]))
                core, ctx = _crop3(vol, c, (2, 4, 4)), _crop3(vol, c, (4, 16, 16))
                if core is not None and ctx is not None:
                    val = (float(core.mean()) / frame_norm[t], float(ctx.mean()) / frame_norm[t],
                           float(core.max()) / frame_norm[t])
            except Exception:
                pass
        inten_cache[nid] = val
        return val

    T = max(1, int(cache.T) - 1)
    rows_s, rows_c1, rows_q, rows_x, feats = [], [], [], [], []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        # sources with exactly one child in the next frame; flow prediction, neighbours and density batched per frame
        srcs = [s for s in by_t[t] if len(children.get(s, ())) == 1 and nodes[children[s][0]]["t"] == t + 1]
        if not srcs:
            continue
        PS = np.stack([pos[s] for s in srcs])
        PRED = flow.forward_many(t, PS, 1) if flow is not None else PS
        tree_next = cKDTree(np.stack([pos[i] for i in nxt]))
        tree_now = cKDTree(np.stack([pos[i] for i in by_t[t]]))
        near_all = tree_next.query_ball_point(PRED, cfg.div_radius_um)
        dens_all = tree_now.query_ball_point(PS, 10.0, return_length=True) - 1
        pnear_all = None
        if pseudo_by_t.get(t + 1):
            pq_ids = pseudo_by_t[t + 1]
            pnear_all = cKDTree(np.stack([pos[q] for q in pq_ids])).query_ball_point(PRED, cfg.div_radius_um)
        va = vb = None
        for si, s in enumerate(srcs):
            c1 = children[s][0]
            ps, pred = PS[si], PRED[si]
            qs = [nxt[i] for i in near_all[si] if nxt[i] != c1]
            if pnear_all is not None:
                qs += [pq_ids[i] for i in sorted(pnear_all[si])]
            if not qs:
                continue
            p_sc1, lg_sc1 = link(s, c1)
            dens = int(dens_all[si])
            Is, Ic = inten(s), inten(c1)
            Ip = inten(parent[s]) if s in parent else (float("nan"), float("nan"), float("nan"))
            s_back, c1_len = run_len(s, False), run_len(c1, True)
            cs = cmap.get(s)
            pc1 = pos[c1]
            v1 = pc1 - pred
            n1 = _n(v1)
            d_sc1 = _n(pc1 - ps)
            for q in qs:
                pq = pos[q]
                v2 = pq - pred
                n2 = _n(v2)
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (n1 * n2))))) if n1 > 1e-6 and n2 > 1e-6 \
                    else float("nan")
                if not ang >= cfg.div_min_angle_deg:     # daughters leave in different directions (GT p10 88 degrees)
                    continue
                x = parent.get(q)
                p_sq, lg_sq = link(s, q)
                best_p, ranks = parents_of(q)
                p_xq, lg_xq = link(x, q) if x is not None else (float("nan"), float("nan"))
                Iq = inten(q)
                g1, g2 = children.get(c1, []), children.get(q, [])
                d_c1q = _n(pc1 - pq)
                div_gap = (_n(pos[g1[0]] - pos[g2[0]]) - d_c1q if len(g1) == 1 and len(g2) == 1 else float("nan"))
                ncc = [float("nan")] * 3
                if frames is not None and cfg.div_ncc:
                    try:
                        if va is None:
                            va, vb = frames.get(t), frames.get(t + 1)
                        zs, zc, zq = (tuple(float((nodes[i] if i in nodes else pseudo[i])[k]) for k in ("z", "y", "x"))
                                      for i in (s, c1, q))
                        ncc = [_ncc_best(vb, zc, vb, zq), _ncc_best(va, zs, vb, zq), _ncc_best(va, zs, vb, zc)]
                    except Exception:
                        pass
                feats.append([
                    d_sc1, _n(pq - ps), d_c1q,
                    n1, n2, ang, _n((pc1 + pq) / 2 - pred),
                    abs(n1 - n2) / max((n1 + n2) / 2, 1e-6), abs(float(pc1[0] - pq[0])),
                    float(x is not None), _n(pos[x] - pq) if x is not None else float("nan"),
                    float(len(children.get(x, []))) if x is not None else float("nan"),
                    float(run_len(q, True)), float(c1_len), float(s_back),
                    float(run_len(x, False)) if x is not None else float("nan"),
                    p_sq, lg_sq, p_sc1, lg_sc1, best_p, float(ranks.get(cs, 99)) if cs is not None else 99.0,
                    p_xq, lg_xq, cos(s, q), cos(s, c1), cos(c1, q), ncc[0], ncc[1], ncc[2],
                    cval(cache.prob, s), cval(cache.prob, c1), cval(cache.prob, q),
                    dcv(s), dcv(c1), dcv(q),
                    div_gap, float(dens), float(len(ranks)), t / T,
                    Is[0], Is[0] / max(Is[1], 1e-6), Is[0] / max(Ip[0], 1e-6) if s in parent else float("nan"),
                    Ic[0], Iq[0], Ic[0] / max(Iq[0], 1e-6), Iq[0] / max(Is[0], 1e-6), Ic[0] / max(Is[0], 1e-6),
                    float(q < 0), Is[2], Ic[2], Iq[2],
                    mitv(s, 1), mitv(s, 2), mitv(c1, 1), mitv(c1, 2), mitv(q, 1), mitv(q, 2),
                    mitv(x, 1) if x is not None else float("nan"), mitv(x, 2) if x is not None else float("nan")])
                rows_s.append(s)
                rows_c1.append(c1)
                rows_q.append(q)
                rows_x.append(-1 if x is None else x)
    X = np.array(feats, dtype=np.float64).reshape(-1, len(DIV_FEATURES))
    return {"s": np.array(rows_s, dtype=np.int64), "c1": np.array(rows_c1, dtype=np.int64),
            "q": np.array(rows_q, dtype=np.int64), "x": np.array(rows_x, dtype=np.int64), "X": X, "pseudo": pseudo}


def save_division_dump(path: Path, nodes: dict, cand: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = {**nodes, **cand.get("pseudo", {})}
    ids = np.array(sorted(nodes), dtype=np.int64)
    np.savez_compressed(path, node_id=ids, node_t=np.array([nodes[i]["t"] for i in ids], dtype=np.int64),
                        node_um=np.stack([point_um(nodes[i]) for i in ids]) if len(ids) else np.zeros((0, 3)),
                        s=cand["s"], c1=cand["c1"], q=cand["q"], x=cand["x"], X=cand["X"],
                        features=np.array(DIV_FEATURES))


# =============================================================================================
# Graph repair
# =============================================================================================

class Repair:
    """Post-ILP graph repair. Nodes: {id: {node_id, t, z, y, x}}; edges: [{source_id, target_id, edge_prob,...}]."""

    def __init__(self, cfg: Config, frames: FrameCache | None, deepcenter: DeepCenter | None,
                 cache: RichCache | None = None):
        self.cfg = cfg
        self.frames = frames
        self.dc = deepcenter
        self.cache = cache
        self.flow: Flow | None = None
        self.stats: dict[str, int] = {}

    def _count(self, key: str, n: int = 1):
        self.stats[key] = self.stats.get(key, 0) + n

    def gate_scale(self, t: int) -> float:
        """Gates of the transition t -> t+1 double when the previous transition was a frozen (identical) frame pair."""
        if not self.cfg.frozen_frames or self.cache is None:
            return 1.0
        return 2.0 if 0 <= t - 1 < len(self.cache.frozen) and self.cache.frozen[t - 1] else 1.0

    def predict_um(self, t: int, p: np.ndarray, steps: int = 1) -> np.ndarray:
        return self.flow.forward(t, p, steps) if self.flow is not None else np.array(p, dtype=np.float64)

    def predict_back_um(self, t: int, p: np.ndarray, steps: int = 1) -> np.ndarray:
        return self.flow.backward(t, p, steps) if self.flow is not None else np.array(p, dtype=np.float64)

    def dc_accept(self, t: int, point: tuple, threshold: float) -> bool:
        if not self.cfg.deepcenter or self.dc is None or self.frames is None:
            return True
        return self.dc.accept(self.frames, t, point, threshold)

    # ---- synthetic node placement ----
    def refine(self, t: int, point: tuple) -> tuple:
        cfg = self.cfg
        if not cfg.refine_synthetic or self.frames is None:
            return point
        try:
            frame = self.frames.get(t)
            z, y, x = (int(round(v)) for v in point)
            z0, z1 = max(0, z - cfg.refine_win_z), min(frame.shape[0], z + cfg.refine_win_z + 1)
            y0, y1 = max(0, y - cfg.refine_win_yx), min(frame.shape[1], y + cfg.refine_win_yx + 1)
            x0, x1 = max(0, x - cfg.refine_win_yx), min(frame.shape[2], x + cfg.refine_win_yx + 1)
            patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
            if patch.size == 0:
                return point
            w = np.maximum(patch - float(np.percentile(patch, 20.0)), 0.0)
            total = float(w.sum())
            if total <= 0:
                return point
            refined = (float((w * np.arange(z0, z1)[:, None, None]).sum() / total),
                       float((w * np.arange(y0, y1)[None, :, None]).sum() / total),
                       float((w * np.arange(x0, x1)[None, None, :]).sum() / total))
            if dist_points_um(refined, point) > cfg.refine_max_shift_um:
                return point
            self._count("refined_synthetic")
            return refined
        except Exception:
            return point

    # ---- cached weak peaks ----
    def weak_snap(self, t: int, point: tuple) -> tuple | None:
        """A cached weak peak near `point` (voxels) at frame t, if the config allows snapping to one."""
        cfg = self.cfg
        if not cfg.gap_use_weak or self.cache is None:
            return None
        p_um = np.array(point, dtype=np.float64) * np.array(SCALE_UM)
        ids, _ = self.cache.near(t, p_um, cfg.rescue_gate_um, "weak")
        for i in ids:
            if self.cache.prob[i] >= cfg.rescue_min_prob:
                return tuple(float(v) for v in self.cache.txyz[i, 1:])
        return None

    def rescue_weak(self, nodes: dict, edges: list) -> tuple[dict, list]:
        """Predict -> look -> accept: extend track ends forward and track starts backward into cached weak peaks.

        At each step the flow-predicted position is checked first for an existing unlinked node of the next frame
        (then the track just continues there), otherwise for a cached weak peak with enough confidence and no other
        node nearby (then the peak becomes a new node). Both directions run up to `rescue_max_steps` new nodes.
        """
        from scipy.spatial import cKDTree
        cfg = self.cfg
        if not cfg.rescue_weak or self.cache is None or not nodes:
            return nodes, edges
        budget = int(round(len(nodes) * cfg.rescue_max_frac))
        outgoing = {e["source_id"] for e in edges}
        incoming = {e["target_id"] for e in edges}
        pred_of: dict[int, int] = {e["target_id"]: e["source_id"] for e in edges}
        succ_of: dict[int, int] = {e["source_id"]: e["target_id"] for e in edges}
        by_t: dict[int, list[int]] = {}
        for nid, n in nodes.items():
            by_t.setdefault(int(n["t"]), []).append(nid)
        trees: dict[int, tuple] = {}

        def frame_tree(t: int):
            if t not in trees:
                ids = by_t.get(t, [])
                trees[t] = (cKDTree(np.stack([point_um(nodes[i]) for i in ids])) if ids else None, ids)
            return trees[t]

        def invalidate(t: int):
            trees.pop(t, None)

        next_id = max(nodes) + 1
        added = linked = dropped_used = 0
        T = self.cache.T

        def extend(start_id: int, direction: int):
            nonlocal next_id, added, linked, dropped_used
            cur = start_id
            steps = 0
            while steps < cfg.rescue_max_steps and added < budget:
                n = nodes[cur]
                t, t2 = int(n["t"]), int(n["t"]) + direction
                if t2 < 0 or t2 >= T:
                    return
                p = point_um(n)
                own = None
                other = pred_of.get(cur) if direction > 0 else succ_of.get(cur)
                if other is not None and other in nodes:
                    own = p - point_um(nodes[other])              # residual own velocity (already in track order)
                    if self.flow is not None:
                        own = own - (self.flow.at(t - 1, point_um(nodes[other])) if direction > 0 else -self.flow.at(t, p))
                base = self.predict_um(t, p, 1) if direction > 0 else self.predict_back_um(t, p, 1)
                pred = base + (cfg.relink_flow_own * own if own is not None else 0.0)
                gate = cfg.rescue_gate_um * (self.gate_scale(t) if direction > 0 else self.gate_scale(t2))
                tree, ids = frame_tree(t2)
                if tree is not None:
                    hits = tree.query_ball_point(pred, gate)
                    free = [ids[h] for h in hits if (ids[h] not in incoming if direction > 0 else ids[h] not in outgoing)]
                    if free:
                        j = min(free, key=lambda i: float(np.linalg.norm(point_um(nodes[i]) - pred)))
                        s_id, g_id = (cur, j) if direction > 0 else (j, cur)
                        edges.append({"source_id": s_id, "target_id": g_id, "edge_prob": None,
                                      "distance_um": dist_um(nodes[s_id], nodes[g_id])})
                        outgoing.add(s_id)
                        incoming.add(g_id)
                        succ_of[s_id], pred_of[g_id] = g_id, s_id
                        linked += 1
                        # joined another track: go on only if the joined node has no edge in the travel direction
                        # (an isolated node); otherwise its own track takes over and extending further would give it
                        # a second parent (backwards) or a third child (forwards)
                        if (direction > 0 and j in outgoing) or (direction < 0 and j in incoming):
                            return
                        cur = j
                        continue
                    if hits:
                        return                                  # an occupied position: do not add a duplicate
                # candidates: weak peaks, plus strong detections that are absent from the graph when asked to
                w_ids, w_d = self.cache.near(t2, pred, gate, "all" if cfg.rescue_dropped else "weak")
                pick = None
                for i, di in zip(w_ids, w_d):
                    if self.cache.prob[i] < cfg.rescue_min_prob:
                        continue
                    if tree is not None and tree.query_ball_point(self.cache.um[i], cfg.rescue_clearance_um):
                        continue
                    pick = i
                    break
                if pick is None:
                    return
                z, y, x = (float(v) for v in self.cache.txyz[pick, 1:])
                nodes[next_id] = {"node_id": next_id, "t": t2, "z": z, "y": y, "x": x, "synthetic": 1, "weak": 1}
                if pick < self.cache.n_strong:
                    dropped_used += 1
                by_t.setdefault(t2, []).append(next_id)
                invalidate(t2)
                s_id, g_id = (cur, next_id) if direction > 0 else (next_id, cur)
                edges.append({"source_id": s_id, "target_id": g_id, "edge_prob": None,
                              "distance_um": dist_um(nodes[s_id], nodes[g_id])})
                outgoing.add(s_id)
                incoming.add(g_id)
                succ_of[s_id], pred_of[g_id] = g_id, s_id
                cur = next_id
                next_id += 1
                added += 1
                steps += 1

        ends = [i for i in sorted(nodes) if i not in outgoing and i in incoming]
        starts = [i for i in sorted(nodes) if i not in incoming and i in outgoing]
        for i in ends:
            if i not in outgoing:
                extend(i, +1)
        for i in starts:
            if i not in incoming:
                extend(i, -1)
        self._count("rescue_weak_nodes", added)
        self._count("rescue_dropped_strong", dropped_used)
        self._count("rescue_linked_existing", linked)
        return nodes, edges

    # ---- 1. edge sanity ----
    def clean_edges(self, nodes: dict, raw_edges: list) -> list:
        out = []
        for e in raw_edges:
            s, t = nodes.get(e["source_id"]), nodes.get(e["target_id"])
            if s is None or t is None or t["t"] != s["t"] + 1:
                continue
            e = dict(e)
            e["distance_um"] = dist_um(s, t)
            if self.cfg.edge_max_um > 0 and e["distance_um"] > self.cfg.edge_max_um:
                continue
            out.append(e)
        return out

    # ---- 2. motion relink (replaces ILP edges; ILP probability is a bonus) ----
    def motion_relink(self, nodes: dict, edges: list) -> list:
        from scipy.optimize import linear_sum_assignment
        cfg = self.cfg
        probs: dict[tuple[int, int], float] = {}
        for e in edges:
            p = e.get("edge_prob")
            if p is None or not np.isfinite(float(p)):
                continue
            key = (e["source_id"], e["target_id"])
            probs[key] = max(probs.get(key, -np.inf), float(p))
        ids_by_t: dict[int, list[int]] = {}
        for nid, n in nodes.items():
            ids_by_t.setdefault(n["t"], []).append(nid)
        for ids in ids_by_t.values():
            ids.sort()
        if ids_by_t and max(len(v) for v in ids_by_t.values()) > cfg.relink_max_frame_nodes:
            self._count("relink_skipped_large_frame")
            return edges
        pos = {nid: point_um(n) for nid, n in nodes.items()}
        pred_pos: dict[int, np.ndarray] = {}
        probs_by_src: dict[int, dict[int, float]] = {}
        for (s, t), p in probs.items():
            if p < 0.0 or p > 1.0:
                p = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, p))))
            probs_by_src.setdefault(s, {})[t] = float(np.clip(p, 0.0, 1.0))

        # ILP division forks (a source with two targets in the next frame) survive the relink when asked to
        forks: dict[int, list[dict]] = {}
        if cfg.relink_keep_forks:
            by_src: dict[int, list[dict]] = {}
            for e in edges:
                s, g = nodes.get(e["source_id"]), nodes.get(e["target_id"])
                if s is not None and g is not None and g["t"] == s["t"] + 1:
                    by_src.setdefault(e["source_id"], []).append(e)
            forks = {s: es[:2] for s, es in by_src.items() if len(es) >= 2}
        use_flow = cfg.relink_flow and self.flow is not None

        def assign(src: list[int], tgt: list[int], gate: float, t: int):
            if not src or not tgt:
                return []
            S = np.stack([pos[i] for i in src])
            Tg = np.stack([pos[j] for j in tgt])
            prev = np.stack([pred_pos.get(i, pos[i]) for i in src])
            has_prev = np.array([i in pred_pos for i in src])
            if use_flow:
                U = np.stack([self.flow.at(t, S[r], own=S[r]) for r in range(len(src))])
                Uprev = np.stack([self.flow.at(t - 1, prev[r]) if has_prev[r] else np.zeros(3) for r in range(len(src))])
                own = np.where(has_prev[:, None], (S - prev) - Uprev, 0.0)
                predicted = S + U + cfg.relink_flow_own * own
            else:
                predicted = np.where(has_prev[:, None], S + cfg.relink_velocity * (S - prev), S)
            raw = np.sqrt(((Tg[None, :, :] - S[:, None, :]) ** 2).sum(-1))
            motion = np.sqrt(((Tg[None, :, :] - predicted[:, None, :]) ** 2).sum(-1))
            P = np.zeros_like(raw)
            col = {j: c for c, j in enumerate(tgt)}
            for r, i in enumerate(src):
                for j, p in probs_by_src.get(i, {}).items():
                    c = col.get(j)
                    if c is not None:
                        P[r, c] = p
            big = gate * 1000.0 + 1.0
            reach = raw if not use_flow else np.minimum(raw, motion)   # with a flow prediction the gate follows it
            cost = np.where(reach <= gate, motion + cfg.relink_raw_weight * raw - cfg.relink_bonus * P, big)
            rows, cols = linear_sum_assignment(cost)
            return [(src[r], tgt[c], float(raw[r, c]), float(P[r, c])) for r, c in zip(rows, cols) if cost[r, c] < big]

        # ---- public (Harmonic Fusion V3) relink: seed assignment -> k-median flow of the seed matches -> tight/relaxed
        from scipy.spatial import cKDTree

        def pub_field(matches: list):
            """Median displacement of the k nearest seed matches within the radius, or None (public _flow_predictor)."""
            if len(matches) < cfg.pub_flow_min_samples:
                return None
            fsrc = np.stack([pos[s] for s, _ in matches])
            fdisp = np.stack([pos[g] - pos[s] for s, g in matches])
            tree = cKDTree(fsrc)
            kq = min(cfg.pub_flow_k + 1, len(fsrc))

            def predict(p, exclude):
                d, idx = tree.query(p, k=kq, distance_upper_bound=cfg.pub_flow_radius_um)
                d, idx = np.atleast_1d(d), np.atleast_1d(idx)
                keep = np.isfinite(d) & (d >= exclude)
                idx = idx[keep][:cfg.pub_flow_k]
                return None if idx.size == 0 else np.median(fdisp[idx], axis=0)
            return predict

        def assign_pub(src: list[int], tgt: list[int], gate: float, field, exclude: float):
            if not src or not tgt:
                return []
            S = np.stack([pos[i] for i in src])
            Tg = np.stack([pos[j] for j in tgt])
            predicted = S.copy()
            for r, i in enumerate(src):
                step = field(S[r], exclude) if field is not None else None
                if step is not None:
                    predicted[r] = S[r] + step
                elif i in pred_pos:
                    predicted[r] = S[r] + cfg.relink_velocity * (S[r] - pred_pos[i])
            raw = np.sqrt(((Tg[None, :, :] - S[:, None, :]) ** 2).sum(-1))
            motion = np.sqrt(((Tg[None, :, :] - predicted[:, None, :]) ** 2).sum(-1))
            P = np.zeros_like(raw)
            col = {j: c for c, j in enumerate(tgt)}
            for r, i in enumerate(src):
                for j, p in probs_by_src.get(i, {}).items():
                    c = col.get(j)
                    if c is not None:
                        P[r, c] = p
            big = gate * 1000.0 + 1.0
            admit = (raw <= gate) | (motion <= gate) if field is not None else (raw <= gate)
            cost = np.where(admit, motion - cfg.relink_bonus * P, big)
            rows, cols = linear_sum_assignment(cost)
            return [(src[r], tgt[c], float(raw[r, c]), float(P[r, c])) for r, c in zip(rows, cols) if cost[r, c] < big]

        pub = cfg.relink_mode == "public"
        prev_field = None
        out = []
        for t in sorted(ids_by_t):
            src, tgt = ids_by_t.get(t, []), ids_by_t.get(t + 1, [])
            if not src or not tgt:
                continue
            free_s, free_t = set(src), set(tgt)
            for s_id in [i for i in src if i in forks]:
                for e in forks[s_id]:
                    if e["target_id"] in free_t:
                        out.append({"source_id": s_id, "target_id": e["target_id"], "edge_prob": e.get("edge_prob"),
                                    "distance_um": dist_um(nodes[s_id], nodes[e["target_id"]])})
                        free_t.discard(e["target_id"])
                        pred_pos[e["target_id"]] = pos[s_id]
                        self._count("relink_kept_fork_edges")
                free_s.discard(s_id)
            scale = self.gate_scale(t)
            if pub:
                seed = assign_pub([i for i in src if i in free_s], [j for j in tgt if j in free_t],
                                  cfg.pub_seed_gate_um * scale, prev_field, 0.0)
                field = pub_field([(s, d) for s, d, _, _ in seed])
                if field is not None:
                    self._count("relink_pub_flow_frames")
                gates = (cfg.pub_tight_um, cfg.relink_relaxed_um) if field is not None else \
                    (cfg.pub_seed_gate_um, cfg.relink_relaxed_um)
                exclude = cfg.pub_flow_exclude_um if field is not None else 0.0
                matches = []
                for gate in gates:
                    for s, d, raw, p in assign_pub([i for i in src if i in free_s], [j for j in tgt if j in free_t],
                                                   gate * scale, field, exclude):
                        free_s.discard(s)
                        free_t.discard(d)
                        out.append({"source_id": s, "target_id": d, "edge_prob": p, "distance_um": raw})
                        pred_pos[d] = pos[s]
                        matches.append((s, d))
                prev_field = pub_field(matches)
                continue
            tight, relaxed = cfg.relink_tight_um, cfg.relink_relaxed_um
            if cfg.relink_dense_per_frame > 0 and len(src) >= cfg.relink_dense_per_frame:
                tight, relaxed = cfg.relink_dense_tight_um, cfg.relink_dense_relaxed_um
                self._count("relink_dense_frames")
            for gate in (tight * scale, relaxed * scale):
                for s, d, raw, p in assign([i for i in src if i in free_s], [j for j in tgt if j in free_t], gate, t):
                    free_s.discard(s)
                    free_t.discard(d)
                    out.append({"source_id": s, "target_id": d, "edge_prob": p, "distance_um": raw})
                    pred_pos[d] = pos[s]
        return out if out else edges

    @staticmethod
    def single_parent(edges: list) -> list:
        best: dict[int, dict] = {}
        for e in edges:
            cur = best.get(e["target_id"])
            if cur is None or edge_rank_key(e) > edge_rank_key(cur):
                best[e["target_id"]] = e
        keep = {id(e) for e in best.values()}
        return [e for e in edges if id(e) in keep]

    # ---- 3. single-frame gaps ----
    def close_gaps(self, nodes: dict, edges: list) -> tuple[dict, list]:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        cfg = self.cfg
        if not cfg.gap_close or not edges:
            return nodes, edges
        outgoing = {e["source_id"] for e in edges}
        incoming = {e["target_id"] for e in edges}
        incident = outgoing | incoming
        ends_by_t, starts_by_t, isolated_by_t, all_by_t = {}, {}, {}, {}
        for nid, n in nodes.items():
            t = n["t"]
            all_by_t.setdefault(t, []).append(nid)
            if nid not in outgoing:
                ends_by_t.setdefault(t, []).append(nid)
            if nid not in incoming:
                starts_by_t.setdefault(t, []).append(nid)
            if nid not in incident:
                isolated_by_t.setdefault(t, []).append(nid)
        max_synth = min(cfg.gap_max_added_abs, max(1, int(round(len(nodes) * cfg.gap_max_added_frac)))
                        if cfg.gap_max_added_frac > 0 else 0)
        next_id = max(nodes) + 1 if nodes else 1
        used_starts, used_isolated = set(), set()
        n_synth = 0
        new_edges = []
        spacing_cache: dict[int, dict[int, float]] = {}

        def spacing(t: int) -> dict[int, float]:
            if t not in spacing_cache:
                ids = all_by_t.get(t, [])
                if len(ids) <= 1:
                    spacing_cache[t] = {i: cfg.gap_density_reference_um for i in ids}
                else:
                    P = np.stack([point_um(nodes[i]) for i in ids])
                    k = min(len(ids), max(2, cfg.gap_density_neighbors + 1))
                    d, _ = cKDTree(P).query(P, k=k)
                    d = d[:, None] if d.ndim == 1 else d
                    spacing_cache[t] = {i: float(np.median(d[r, 1:][np.isfinite(d[r, 1:])]))
                                        if np.isfinite(d[r, 1:]).any() else cfg.gap_density_reference_um
                                        for r, i in enumerate(ids)}
            return spacing_cache[t]

        gap = 1
        use_flow = cfg.gap_flow and self.flow is not None
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [s for s in starts_by_t.get(t + gap + 1, []) if s not in used_starts]
            if not end_ids or not start_ids:
                continue
            base = cfg.gap_um * (gap + 1) * self.gate_scale(t)
            if use_flow:
                # residual between the end advected forward and the start, and the start advected backward and the end
                fwd = {e: self.predict_um(t, point_um(nodes[e]), gap + 1) for e in end_ids}
                bwd = {s: self.predict_back_um(t + gap + 1, point_um(nodes[s]), gap + 1) for s in start_ids}
                d = np.array([[0.5 * (float(np.linalg.norm(fwd[e] - point_um(nodes[s])))
                                      + float(np.linalg.norm(bwd[s] - point_um(nodes[e])))) for s in start_ids]
                              for e in end_ids])
            else:
                d = np.array([[dist_points_um(zyx(nodes[e]), zyx(nodes[s])) for s in start_ids] for e in end_ids])
            thr = np.full_like(d, base)
            if cfg.gap_density_adaptive:
                sp_src, sp_tgt = spacing(t), spacing(t + gap + 1)
                for i, e in enumerate(end_ids):
                    for j, s in enumerate(start_ids):
                        local = 0.5 * (sp_src.get(e, cfg.gap_density_reference_um)
                                       + sp_tgt.get(s, cfg.gap_density_reference_um))
                        delta = float(np.clip(cfg.gap_density_gain * (local - cfg.gap_density_reference_um),
                                              -cfg.gap_density_max_step_delta_um, cfg.gap_density_max_step_delta_um))
                        thr[i, j] = base + delta * (gap + 1)
            allowed = d <= thr
            big = float(np.max(thr)) * 1000.0 + 1.0
            rows, cols = linear_sum_assignment(np.where(allowed, d, big))
            for r, c in zip(rows, cols):
                if not allowed[r, c]:
                    continue
                src_id, tgt_id = end_ids[int(r)], start_ids[int(c)]
                if src_id in outgoing or tgt_id in used_starts:
                    continue
                src, tgt = nodes[src_id], nodes[tgt_id]
                mid_t = src["t"] + gap
                if use_flow:
                    mid_um = 0.5 * (self.predict_um(t, point_um(src), 1) + self.predict_back_um(t + 2, point_um(tgt), 1))
                    mid = tuple(float(v) for v in (mid_um / np.array(SCALE_UM)))
                else:
                    mid = tuple((a + b) / 2.0 for a, b in zip(zyx(src), zyx(tgt)))
                mid_id, reused = None, False
                cands = [i for i in isolated_by_t.get(mid_t, []) if i not in used_isolated]
                if cands:
                    dd = [dist_points_um(zyx(nodes[i]), mid) for i in cands]
                    k = int(np.argmin(dd))
                    if dd[k] <= cfg.gap_reuse_um:
                        mid_id, reused = cands[k], True
                if mid_id is None:
                    if n_synth >= max_synth:
                        self._count("gap_skipped_cap")
                        continue
                    p, weak_hit = self.weak_snap(mid_t, mid), False
                    if p is not None:
                        weak_hit = True
                    else:
                        p = self.refine(mid_t, mid)
                    if d[r, c] >= cfg.deepcenter_gap_min_span_um and not self.dc_accept(
                            mid_t, p, cfg.deepcenter_gap_threshold):
                        self._count("gap_rejected_deepcenter")
                        continue
                    mid_id = next_id
                    next_id += 1
                    nodes[mid_id] = {"node_id": mid_id, "t": mid_t, "z": p[0], "y": p[1], "x": p[2],
                                     "synthetic": 1}
                    n_synth += 1
                    if weak_hit:
                        self._count("gap_weak_snapped")
                if reused:
                    used_isolated.add(mid_id)
                mid_node = nodes[mid_id]
                new_edges += [
                    {"source_id": src_id, "target_id": mid_id, "edge_prob": None, "distance_um": dist_um(src, mid_node)},
                    {"source_id": mid_id, "target_id": tgt_id, "edge_prob": None, "distance_um": dist_um(mid_node, tgt)},
                ]
                outgoing.update((src_id, mid_id))
                incoming.update((mid_id, tgt_id))
                used_starts.add(tgt_id)
                self._count("gap_closed")
        self._count("gap_synthetic_nodes", n_synth)
        return nodes, edges + new_edges

    # ---- 4. two-frame gaps (strict, with motion context) ----
    def close_gap2(self, nodes: dict, edges: list) -> tuple[dict, list]:
        cfg = self.cfg
        if not cfg.gap2 or not edges or not nodes:
            return nodes, edges
        outgoing = {e["source_id"] for e in edges}
        incoming = {e["target_id"] for e in edges}
        by_src, by_tgt = {}, {}
        for e in edges:
            by_src.setdefault(e["source_id"], []).append(e["target_id"])
            by_tgt.setdefault(e["target_id"], []).append(e["source_id"])
        successor = {k: v[0] for k, v in by_src.items() if len(v) == 1}
        predecessor = {k: v[0] for k, v in by_tgt.items() if len(v) == 1}
        ends_by_t, starts_by_t = {}, {}
        for nid, n in nodes.items():
            if nid not in outgoing:
                ends_by_t.setdefault(n["t"], []).append(nid)
            if nid not in incoming:
                starts_by_t.setdefault(n["t"], []).append(nid)
        cap = min(cfg.gap2_max_links_abs, max(1, int(round(len(edges) * cfg.gap2_max_links_frac))))
        pos = lambda i: point_um(nodes[i])  # noqa: E731
        use_flow = cfg.gap_flow and self.flow is not None
        proposals = []
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = starts_by_t.get(t + 3, [])
            scale = self.gate_scale(t)
            for e_id in end_ids:
                pe = pos(e_id)
                pe_fwd = self.predict_um(t, pe, 3) if use_flow else pe
                for s_id in start_ids:
                    ps = pos(s_id)
                    if use_flow:
                        # residual after advecting the end forward and the start backward through the flow
                        dist = 0.5 * (float(np.linalg.norm(ps - pe_fwd))
                                      + float(np.linalg.norm(self.predict_back_um(t + 3, ps, 3) - pe)))
                    else:
                        dist = float(np.linalg.norm(ps - pe))
                    if dist > cfg.gap2_max_total_um * scale or dist / 3.0 > cfg.gap2_max_step_um * scale:
                        continue
                    step = (ps - pe) / 3.0
                    ok, penalty = False, 0.0
                    for other, sign in ((predecessor.get(e_id), -1), (successor.get(s_id), 1)):
                        if other is None:
                            continue
                        ctx = (pe - pos(other)) if sign < 0 else (pos(other) - ps)
                        cn, sn = float(np.linalg.norm(ctx)), float(np.linalg.norm(step))
                        if cn <= 0.01 or sn <= 0.01:
                            ok = True
                            continue
                        cos = float(np.dot(ctx, step) / (cn * sn + 1e-9))
                        if cos > -0.25 and np.linalg.norm(ctx - step) <= 6.0:
                            ok = True
                        penalty += max(0.0, 0.25 - cos)
                    if ok:
                        proposals.append((dist + 2.0 * penalty, e_id, s_id, t))
        proposals.sort(key=lambda p: p[0])
        selected, used_e, used_s, per_frame = [], set(), set(), {}
        for p in proposals:
            if len(selected) >= cap:
                break
            _, e_id, s_id, t = p
            if e_id in used_e or s_id in used_s:
                continue
            if per_frame.get(t, 0) >= max(1, int(round(len(ends_by_t.get(t, [])) * cfg.gap2_frame_frac_cap))):
                continue
            selected.append(p)
            used_e.add(e_id)
            used_s.add(s_id)
            per_frame[t] = per_frame.get(t, 0) + 1
        next_id = max(nodes) + 1
        new_edges = []
        for _, e_id, s_id, _ in selected:
            src, tgt, prev = nodes[e_id], nodes[s_id], e_id
            for k in (1, 2):
                frac = k / 3.0
                if use_flow:
                    mid_um = ((1.0 - frac) * self.predict_um(src["t"], point_um(src), k)
                              + frac * self.predict_back_um(tgt["t"], point_um(tgt), 3 - k))
                    mid = tuple(float(v) for v in (mid_um / np.array(SCALE_UM)))
                else:
                    mid = tuple(a + (b - a) * frac for a, b in zip(zyx(src), zyx(tgt)))
                p = self.weak_snap(src["t"] + k, mid)
                if p is None:
                    p = self.refine(src["t"] + k, mid)
                else:
                    self._count("gap2_weak_snapped")
                nodes[next_id] = {"node_id": next_id, "t": src["t"] + k, "z": p[0], "y": p[1], "x": p[2],
                                  "synthetic": 1}
                new_edges.append({"source_id": prev, "target_id": next_id, "edge_prob": None,
                                  "distance_um": dist_um(nodes[prev], nodes[next_id])})
                prev = next_id
                next_id += 1
            new_edges.append({"source_id": prev, "target_id": s_id, "edge_prob": None,
                              "distance_um": dist_um(nodes[prev], tgt)})
        self._count("gap2_closed", len(selected))
        return nodes, edges + new_edges

    # ---- 4b. tracklet stitching with the flow and the neighbour constellation as identity ----
    def stitch_tracks(self, nodes: dict, edges: list) -> tuple[dict, list]:
        """Link remaining track ends to track starts across 1..stitch_max_gap missing frames.

        Cost = mean flow residual (end advected forward, start advected backward) + constellation mismatch
        (Chamfer distance between the relative positions of the k nearest neighbours of both nodes) + optional
        feature distance. Greedy by cost, both nodes must be free; the missing frames are filled along the flow,
        snapped to cached weak peaks when available, vetoed by DeepCenter like ordinary gap nodes.
        """
        from scipy.spatial import cKDTree
        cfg = self.cfg
        if not cfg.stitch or not edges or not nodes:
            return nodes, edges
        outgoing = {e["source_id"] for e in edges}
        incoming = {e["target_id"] for e in edges}
        by_t: dict[int, list[int]] = {}
        for nid, n in nodes.items():
            by_t.setdefault(int(n["t"]), []).append(nid)
        T = max(by_t) + 1
        trees = {t: (cKDTree(np.stack([point_um(nodes[i]) for i in ids])), ids) for t, ids in by_t.items()}

        def constellation(nid: int) -> np.ndarray:
            n = nodes[nid]
            tree, ids = trees[int(n["t"])]
            p = point_um(n)
            k = min(cfg.stitch_k + 1, len(ids))
            d, j = tree.query(p, k=k)
            d, j = np.atleast_1d(d), np.atleast_1d(j)
            offs = [point_um(nodes[ids[jj]]) - p for dd, jj in zip(d, j) if dd > 1e-6]
            return np.stack(offs) if offs else np.zeros((0, 3))

        def chamfer(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) == 0 or len(b) == 0:
                return 0.0
            d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
            return 0.5 * (float(d.min(1).mean()) + float(d.min(0).mean()))

        feat = self.cache.feat if (cfg.stitch_feature_weight > 0 and self.cache is not None) else None

        def feat_cost(a: int, b: int) -> float:
            if feat is None or a >= len(feat) or b >= len(feat):
                return 0.0
            fa, fb = feat[a].astype(np.float64), feat[b].astype(np.float64)
            na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
            return 0.0 if na == 0 or nb == 0 else 1.0 - float(fa @ fb / (na * nb))

        ends = [i for i in nodes if i not in outgoing and i in incoming and int(nodes[i]["t"]) < T - 1]
        starts_by_t: dict[int, list[int]] = {}
        for i in nodes:
            if i not in incoming and i in outgoing and int(nodes[i]["t"]) > 0:
                starts_by_t.setdefault(int(nodes[i]["t"]), []).append(i)
        const_cache: dict[int, np.ndarray] = {}
        proposals = []
        for e_id in ends:
            t = int(nodes[e_id]["t"])
            pe = point_um(nodes[e_id])
            for gap in range(1, cfg.stitch_max_gap + 1):
                delta = gap + 1
                cands = starts_by_t.get(t + delta, [])
                if not cands:
                    continue
                fwd = self.predict_um(t, pe, delta)
                gate = cfg.stitch_um_per_step * delta * self.gate_scale(t)
                s_tree = cKDTree(np.stack([point_um(nodes[s]) for s in cands]))
                for k in s_tree.query_ball_point(fwd, gate):
                    s_id = cands[k]
                    ps = point_um(nodes[s_id])
                    bwd = self.predict_back_um(t + delta, ps, delta)
                    resid = 0.5 * (float(np.linalg.norm(ps - fwd)) + float(np.linalg.norm(pe - bwd)))
                    if resid > gate:
                        continue
                    if e_id not in const_cache:
                        const_cache[e_id] = constellation(e_id)
                    if s_id not in const_cache:
                        const_cache[s_id] = constellation(s_id)
                    cost = (resid + cfg.stitch_constellation_weight * chamfer(const_cache[e_id], const_cache[s_id])
                            + 10.0 * cfg.stitch_feature_weight * feat_cost(e_id, s_id))
                    proposals.append((cost, e_id, s_id, delta, fwd, bwd))
        proposals.sort(key=lambda p: p[0])
        cap = max(1, int(round(len(edges) * cfg.stitch_max_frac)))
        used_e, used_s = set(), set()
        next_id = max(nodes) + 1
        new_edges = []
        n_done = 0
        for cost, e_id, s_id, delta, fwd, bwd in proposals:
            if n_done >= cap:
                break
            if e_id in used_e or s_id in used_s:
                continue
            src, tgt = nodes[e_id], nodes[s_id]
            t = int(src["t"])
            chain, ok = [], True
            for k in range(1, delta):
                frac = k / delta
                mid_um = (1.0 - frac) * self.predict_um(t, point_um(src), k) + frac * self.predict_back_um(t + delta, point_um(tgt), delta - k)
                mid = tuple(float(v) for v in (mid_um / np.array(SCALE_UM)))
                p = self.weak_snap(t + k, mid)
                if p is None:
                    p = self.refine(t + k, mid)
                span = float(np.linalg.norm(point_um(tgt) - point_um(src)))
                if span >= cfg.deepcenter_gap_min_span_um and not self.dc_accept(t + k, p, cfg.deepcenter_gap_threshold):
                    ok = False
                    break
                chain.append((t + k, p))
            if not ok:
                self._count("stitch_rejected_deepcenter")
                continue
            prev = e_id
            for tk, p in chain:
                nodes[next_id] = {"node_id": next_id, "t": tk, "z": p[0], "y": p[1], "x": p[2], "synthetic": 1}
                new_edges.append({"source_id": prev, "target_id": next_id, "edge_prob": None,
                                  "distance_um": dist_um(nodes[prev], nodes[next_id])})
                prev = next_id
                next_id += 1
            new_edges.append({"source_id": prev, "target_id": s_id, "edge_prob": None,
                              "distance_um": dist_um(nodes[prev], tgt)})
            used_e.add(e_id)
            used_s.add(s_id)
            n_done += 1
            self._count(f"stitch_gap{delta - 1}")
        self._count("stitched", n_done)
        return nodes, edges + new_edges

    # ---- 5. safe divisions (second daughter from an orphan next to the existing child) ----
    def safe_divisions(self, nodes: dict, edges: list) -> list:
        from scipy.spatial import cKDTree
        cfg = self.cfg
        if not cfg.safe_div or not edges or not nodes:
            return edges
        out_by_src: dict[int, list[dict]] = {}
        incoming: set[int] = set()
        for e in edges:
            out_by_src.setdefault(e["source_id"], []).append(e)
            incoming.add(e["target_id"])
        ids_by_t: dict[int, list[int]] = {}
        for nid, n in nodes.items():
            ids_by_t.setdefault(n["t"], []).append(nid)
        existing = {(e["source_id"], e["target_id"]) for e in edges}
        global_cap = max(1, int(round(max(1, len(edges)) * cfg.safe_div_global_frac_cap)))
        added, used_t, used_s = [], set(), set()
        props_by_t: dict[int, tuple] = {}
        for t in sorted(ids_by_t):
            children = ids_by_t.get(t + 1, [])
            if not children:
                continue
            sources = [i for i in ids_by_t[t] if len(out_by_src.get(i, [])) == 1]
            cands = [i for i in children if i not in incoming and i not in used_t]
            if not sources or not cands:
                continue
            tree = cKDTree(np.stack([point_um(nodes[c]) for c in cands])) if cfg.safe_div_mutual_nn else None
            frame_cap = max(1, int(round(len(sources) * cfg.safe_div_frame_frac_cap)))
            proposals = []
            for s_id in sources:
                src = nodes[s_id]
                c1_id = out_by_src[s_id][0]["target_id"]
                c1 = nodes.get(c1_id)
                if c1 is None or c1["t"] != t + 1:
                    continue
                # with the flow the daughters are measured from where the parent was heading, not where it was
                src_um = self.predict_um(t, point_um(src), 1) if (cfg.safe_div_flow and self.flow is not None) \
                    else point_um(src)
                d_child = float(np.linalg.norm(point_um(c1) - src_um))
                if d_child > cfg.safe_div_existing_child_um:
                    continue
                nn_id = cands[int(tree.query(point_um(c1))[1])] if tree is not None else None
                for q_id in cands:
                    if (s_id, q_id) in existing:
                        continue
                    q = nodes[q_id]
                    d_parent = float(np.linalg.norm(point_um(q) - src_um))
                    if d_parent > cfg.safe_div_parent_um:
                        continue
                    d_sister = dist_um(c1, q)
                    if d_sister > cfg.safe_div_sister_um:
                        continue
                    if tree is not None and q_id != nn_id:
                        continue
                    kind = "orphan"
                    if cfg.safe_div_diverge_um is not None:
                        diverges = False
                        cur1, cur2 = c1_id, q_id
                        for h in range(max(1, cfg.safe_div_horizon_frames)):    # horizon 0 = the next frame only
                            s1, s2 = out_by_src.get(cur1, []), out_by_src.get(cur2, [])
                            g1 = nodes.get(s1[0]["target_id"]) if len(s1) == 1 else None
                            g2 = nodes.get(s2[0]["target_id"]) if len(s2) == 1 else None
                            if g1 is None or g2 is None or g1["t"] != t + 2 + h or g2["t"] != t + 2 + h:
                                break
                            d_desc = dist_um(g1, g2)
                            if d_desc - d_sister >= cfg.safe_div_diverge_um or \
                                    (cfg.safe_div_horizon_frames > 0 and d_desc >= cfg.safe_div_horizon_um):
                                diverges = True
                                break
                            cur1, cur2 = s1[0]["target_id"], s2[0]["target_id"]
                        if not diverges:
                            if cfg.safe_div_nodiverge_cnn_min <= 0:
                                continue
                            kind = "orphan_nd"                 # no divergence evidence: the CNN must carry it
                    gate_fail = False
                    if cfg.safe_div_min_angle_deg > 0:
                        v1, v2 = point_um(c1) - src_um, point_um(q) - src_um
                        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                        if n1 > 1e-6 and n2 > 1e-6:
                            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (n1 * n2)))))
                            if ang < cfg.safe_div_min_angle_deg:
                                if cfg.safe_div_traj_cnn_min <= 0:
                                    self._count("safe_div_rejected_angle")
                                    continue
                                gate_fail = True
                    if not self.dc_accept(q["t"], zyx(q), cfg.deepcenter_safe_div_threshold):
                        self._count("safe_div_rejected_deepcenter")
                        continue
                    if cfg.safe_div_symmetry_tau > 0:
                        if abs(d_child - d_parent) / max((d_child + d_parent) / 2.0, 1e-6) > cfg.safe_div_symmetry_tau:
                            if cfg.safe_div_traj_cnn_min > 0:
                                gate_fail = True
                            elif cfg.safe_div_asym_cnn_min <= 0:
                                continue
                            else:
                                kind = "orphan_asym" if kind == "orphan" else "orphan_nd"   # both missing: the stricter bar
                    if gate_fail:                            # trajectory rescue: sisters keep separating over t+1..t+3
                        def _succ(nid, tt):
                            o = out_by_src.get(nid, [])
                            if len(o) != 1:
                                return None
                            nn = nodes.get(o[0]["target_id"])
                            return nn if nn is not None and nn["t"] == tt else None
                        a2, b2 = _succ(c1_id, t + 2), _succ(q_id, t + 2)
                        a3 = _succ(a2["node_id"], t + 3) if a2 is not None else None
                        b3 = _succ(b2["node_id"], t + 3) if b2 is not None else None
                        if a2 is None or b2 is None or a3 is None or b3 is None:
                            continue
                        d2s, d3s = dist_um(a2, b2), dist_um(a3, b3)
                        if not (d_sister < d2s < d3s and d3s - d_sister >= cfg.safe_div_traj_margin_um):
                            continue
                        kind = "orphan_traj"
                    proposals.append((d_parent + 0.15 * d_sister, s_id, q_id, d_parent, c1_id, kind))
            props_by_t[t] = (proposals, frame_cap)
        self._pseudo: dict[int, dict] = {}
        if cfg.safe_div_steal_min > 0 or cfg.safe_div_dropped_min > 0 or cfg.safe_div_weak_min > 0:
            self._extra_division_proposals(nodes, out_by_src, incoming, ids_by_t, existing, props_by_t)
        # the current parent ("thief") of every linked node and the parentless nodes, for the steal rules that look at the victim
        self._thief_of = {e["target_id"]: e["source_id"] for e in edges}
        self._orphans = {nid for nid in nodes if nid not in incoming}
        self._succ1 = {s_: es[0]["target_id"] for s_, es in out_by_src.items() if len(es) == 1}
        self._steal_alt: dict[tuple, tuple] = {}
        if cfg.safe_div_cnn_min > 0 or self._pseudo or cfg.safe_div_steal_min > 0 or cfg.safe_div_nodiverge_cnn_min > 0 \
                or cfg.safe_div_asym_cnn_min > 0 or cfg.safe_div_traj_cnn_min > 0:
            props_by_t = self._cnn_veto(nodes, props_by_t)
        parent_edge = {e["target_id"]: k for k, e in enumerate(edges)}
        drop: set[int] = set()
        rank = {"orphan": 0, "orphan_nd": 1, "orphan_asym": 1, "orphan_traj": 1, "dropped": 1, "dropped_dc": 1, "weak": 1, "steal": 2}
        for t in sorted(props_by_t):
            proposals, frame_cap = props_by_t[t]
            if cfg.safe_div_kind_priority:              # orphans first, then revived detections, steals last (caps!)
                proposals.sort(key=lambda p: (rank.get(p[5], 3), p[0]))
            else:
                proposals.sort(key=lambda p: p[0])
            n_frame = 0
            for _, s_id, q_id, d_parent, _c1, kind in proposals:
                if len(added) >= global_cap or n_frame >= frame_cap:
                    break
                if q_id in used_t or s_id in used_s:
                    continue
                if kind in ("orphan", "orphan_nd", "orphan_asym", "orphan_traj") and q_id in incoming:
                    continue
                if kind == "steal":
                    if q_id not in parent_edge or parent_edge[q_id] in drop:
                        continue
                    drop.add(parent_edge[q_id])
                    self._count("safe_div_steals")
                    alt = self._steal_alt.get((s_id, q_id))
                    if alt is not None and alt[1] not in used_t and alt[1] in self._orphans and alt[0] not in used_s:
                        x_id, q2 = alt                        # the victim gets its own parentless continuation back
                        added.append({"source_id": x_id, "target_id": q2, "edge_prob": None,
                                      "distance_um": float(np.linalg.norm(point_um(nodes[q2]) - point_um(nodes[x_id])))})
                        used_t.add(q2)
                        self._count("safe_div_steal_relinks")
                used_t.add(q_id)
                if kind in ("dropped", "weak", "dropped_dc"):   # a dropped strong detection / a weak peak becomes the daughter
                    p = self._pseudo[q_id]
                    nid = max(nodes) + 1
                    nodes[nid] = {"node_id": nid, "t": p["t"], "z": p["z"], "y": p["y"], "x": p["x"]}
                    q_id = nid
                    self._count(f"safe_div_{kind}_daughters")
                added.append({"source_id": s_id, "target_id": q_id, "edge_prob": None, "distance_um": d_parent})
                used_s.add(s_id)
                n_frame += 1
        self._count("safe_divisions", len(added))
        return [e for k, e in enumerate(edges) if k not in drop] + added

    def _extra_division_proposals(self, nodes, out_by_src, incoming, ids_by_t, existing, props_by_t) -> None:
        """Second daughters the orphan-only rule cannot see: nodes already linked to another parent ("steal") and
        strong detections missing from the graph ("dropped"). Same distance, angle, symmetry and DeepCenter gates;
        the cell-state CNN veto decides (with its own, stricter thresholds)."""
        from scipy.spatial import cKDTree
        cfg = self.cfg
        pos = {}
        for t in sorted(ids_by_t):
            children = ids_by_t.get(t + 1, [])
            if not children:
                continue
            sources = [i for i in ids_by_t[t] if len(out_by_src.get(i, [])) == 1]
            if not sources:
                continue
            extra = []
            if cfg.safe_div_steal_min > 0:
                extra += [(i, "steal") for i in children if i in incoming]
            if self.cache is not None and (cfg.safe_div_dropped_min > 0 or cfg.safe_div_weak_min > 0):
                rows = np.arange(len(self.cache.t))
                sel = (self.cache.t == t + 1) & (
                    ((rows < self.cache.n_strong) if cfg.safe_div_dropped_min > 0 else False) |
                    (((rows >= self.cache.n_strong) & (self.cache.prob >= cfg.safe_div_weak_prob)) if cfg.safe_div_weak_min > 0 else False))
                idx = np.nonzero(sel)[0]
                if len(idx):
                    tr = cKDTree(np.stack([point_um(nodes[i]) for i in children]))
                    d, _ = tr.query(self.cache.um[idx])
                    for ci in idx[d > 2.0]:
                        pid = -(int(ci) + 1)
                        self._pseudo[pid] = {"node_id": pid, "t": t + 1, "z": float(self.cache.txyz[ci, 1]),
                                             "y": float(self.cache.txyz[ci, 2]), "x": float(self.cache.txyz[ci, 3])}
                        extra.append((pid, "dropped" if ci < self.cache.n_strong else "weak"))
            if not extra:
                continue
            for i, _ in extra:
                pos[i] = point_um(self._pseudo[i]) if i < 0 else point_um(nodes[i])
            tree = cKDTree(np.stack([pos[i] for i, _ in extra]))
            proposals = props_by_t.get(t, ([], max(1, int(round(len(sources) * cfg.safe_div_frame_frac_cap)))))
            for s_id in sources:
                src = nodes[s_id]
                c1_id = out_by_src[s_id][0]["target_id"]
                c1 = nodes.get(c1_id)
                if c1 is None or c1["t"] != t + 1:
                    continue
                src_um = self.predict_um(t, point_um(src), 1) if (cfg.safe_div_flow and self.flow is not None) \
                    else point_um(src)
                d_child = float(np.linalg.norm(point_um(c1) - src_um))
                if d_child > cfg.safe_div_existing_child_um:
                    continue
                for k in tree.query_ball_point(src_um, cfg.safe_div_parent_um):
                    q_id, kind = extra[k]
                    if q_id == c1_id or (s_id, q_id) in existing:
                        continue
                    q_um = pos[q_id]
                    d_parent = float(np.linalg.norm(q_um - src_um))
                    d_sister = float(np.linalg.norm(point_um(c1) - q_um))
                    if d_sister > cfg.safe_div_sister_um:
                        continue
                    if cfg.safe_div_min_angle_deg > 0:
                        v1, v2 = point_um(c1) - src_um, q_um - src_um
                        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                        if n1 > 1e-6 and n2 > 1e-6:
                            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (n1 * n2)))))
                            if ang < cfg.safe_div_min_angle_deg:
                                continue
                    if cfg.safe_div_symmetry_tau > 0 and \
                            abs(d_child - d_parent) / max((d_child + d_parent) / 2.0, 1e-6) > cfg.safe_div_symmetry_tau:
                        continue
                    qn = self._pseudo[q_id] if q_id < 0 else nodes[q_id]
                    kind_ = kind
                    if not self.dc_accept(t + 1, zyx(qn), cfg.deepcenter_safe_div_threshold):
                        if kind == "dropped" and cfg.safe_div_dropped_dc_min > 0:
                            kind_ = "dropped_dc"              # DeepCenter disagrees: only a strong CNN sum revives it
                        else:
                            continue
                    proposals[0].append((d_parent + 0.15 * d_sister, s_id, q_id, d_parent, c1_id, kind_))
            props_by_t[t] = proposals

    def _cell_state(self, nodes: dict, ids: list) -> dict:
        """(normal, parent, daughter) probabilities of the given nodes: cached per detection in the lab, else the CNN."""
        cfg = self.cfg
        out: dict[int, np.ndarray] = {}
        mit = getattr(self.cache, "mit", None) if self.cache is not None else None
        if mit is not None:
            for i in ids:
                n = nodes[i]
                c, _ = self.cache.near(int(n["t"]), point_um(n), 0.6)
                if len(c) and np.isfinite(mit[c[0]]).all():
                    out[i] = mit[c[0]]
            ids = [i for i in ids if i not in out]          # unscored rows (e.g. weak peaks) fall through to the CNN
            if not ids:
                return out
        if cfg.div_mitosis_model and self.frames is not None and ids:
            import torch
            dev = cuda_device_for_worker()
            txyz = np.array([[nodes[i]["t"], nodes[i]["z"], nodes[i]["y"], nodes[i]["x"]] for i in ids], dtype=np.float64)
            sc = mitosis_score_nodes(mitosis_load(cfg.div_mitosis_model, dev), self.frames.zarr_path, txyz, dev,
                                     norm=cfg.mitosis_norm)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            for i, row in zip(ids, sc):
                if np.isfinite(row).all():
                    out[i] = row
        return out

    def _size_rel(self, nodes: dict, ids: list) -> dict:
        """Nucleus radius relative to the neighbours (cache.rad_rel, lab precomputed) of the given nodes."""
        rr = getattr(self.cache, "rad_rel", None) if self.cache is not None else None
        out: dict[int, float] = {}
        if rr is None:
            return out
        for i in ids:
            n = nodes[i]
            c, _ = self.cache.near(int(n["t"]), point_um(n), 0.6)
            if len(c) and c[0] < len(rr) and np.isfinite(rr[c[0]]):
                out[i] = float(rr[c[0]])
        return out

    def _cnn_veto(self, nodes: dict, props_by_t: dict) -> dict:
        """Drop division proposals whose parent and daughters do not look like a division to the cell-state CNN.
        Orphan daughters need cfg.safe_div_cnn_min; taking a daughter from another track or reviving a dropped
        detection needs the stricter safe_div_steal_min / safe_div_dropped_min, and no score means no fork there."""
        cfg = self.cfg
        allnodes = {**nodes, **getattr(self, "_pseudo", {})}
        thief_of = getattr(self, "_thief_of", {})
        succ1 = getattr(self, "_succ1", {})
        ids = {i for proposals, _ in props_by_t.values() for p in proposals for i in (p[1], p[2], p[4])}
        ids |= {thief_of[p[2]] for proposals, _ in props_by_t.values() for p in proposals if p[5] == "steal" and p[2] in thief_of}
        if cfg.safe_div_temporal_w > 0:                   # mother's predecessor, daughters' successors
            for proposals, _ in props_by_t.values():
                for p in proposals:
                    for i in (thief_of.get(p[1]), succ1.get(p[2]), succ1.get(p[4])):
                        if i is not None and i in allnodes:
                            ids.add(i)
        ids = sorted(ids)
        st = self._cell_state(allnodes, ids)
        # parentless nodes per frame, for the "victim keeps a continuation" steal rule
        orph_by_t: dict[int, tuple] = {}
        if cfg.safe_div_steal_alt_min > 0:
            from scipy.spatial import cKDTree as _KD
            tmp: dict[int, list] = {}
            for nid in getattr(self, "_orphans", set()):
                if nid in nodes:
                    tmp.setdefault(int(nodes[nid]["t"]), []).append(nid)
            for tt, lst in tmp.items():
                orph_by_t[tt] = (lst, _KD(np.stack([point_um(nodes[i]) for i in lst])))

        def victim_alternative(x_id, q_id, t_next):
            """Nearest parentless node at t_next (other than q) within safe_div_steal_alt_um of the victim's predicted position."""
            if t_next not in orph_by_t or x_id not in nodes:
                return None
            lst, tree = orph_by_t[t_next]
            pred = self.predict_um(int(nodes[x_id]["t"]), point_um(nodes[x_id]), 1) if (cfg.safe_div_flow and self.flow is not None) else point_um(nodes[x_id])
            dd, jj = tree.query(pred, k=min(3, len(lst)))
            for d_, j in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                j = int(j)
                if np.isfinite(d_) and j < len(lst) and d_ <= cfg.safe_div_steal_alt_um and lst[j] != q_id:
                    return lst[j]
            return None
        need = {"orphan": cfg.safe_div_cnn_min, "steal": cfg.safe_div_steal_min, "dropped": cfg.safe_div_dropped_min,
                "dropped_dc": max(cfg.safe_div_dropped_dc_min, cfg.safe_div_dropped_min),
                "weak": cfg.safe_div_weak_min, "orphan_nd": max(cfg.safe_div_nodiverge_cnn_min, cfg.safe_div_asym_cnn_min),
                "orphan_asym": cfg.safe_div_asym_cnn_min, "orphan_traj": cfg.safe_div_traj_cnn_min}
        sz = self._size_rel(allnodes, ids) if (cfg.safe_div_size_beta > 0 or cfg.safe_div_size_veto > 0) else {}

        def bonus(i):                                    # nucleus smaller than its neighbours -> more daughter-like
            r = sz.get(i)
            if r is None or cfg.safe_div_size_beta <= 0:
                return 0.0
            return cfg.safe_div_size_beta * max(0.0, min(1.0, (cfg.safe_div_size_ref - r) / max(cfg.safe_div_size_scale, 1e-6)))
        # mother stretch / sister-axis alignment and sister-size rules (lab-precomputed shape and radius caches)
        shp_ax = getattr(self.cache, "shape_axis", None) if self.cache is not None else None
        shp_an = getattr(self.cache, "shape_aniso", None) if self.cache is not None else None
        rad_abs = getattr(self.cache, "rad", None) if self.cache is not None else None
        use_align = cfg.safe_div_align_aniso > 0 and shp_ax is not None
        use_size = cfg.safe_div_sister_size_max > 0 and rad_abs is not None
        if use_size and not sz:
            sz = self._size_rel(allnodes, ids)

        def _row(i):
            n = allnodes[i]
            c, _ = self.cache.near(int(n["t"]), point_um(n), 0.6)
            return int(c[0]) if len(c) else -1

        def aligned(p):
            r = _row(p[1])
            if r < 0 or r >= len(shp_an) or not np.isfinite(shp_an[r]) or shp_an[r] < cfg.safe_div_align_aniso:
                return False
            s = point_um(allnodes[p[4]]) - point_um(allnodes[p[2]])
            n = float(np.linalg.norm(s))
            return n > 1e-6 and abs(float(s @ shp_ax[r])) / n >= cfg.safe_div_align_cos

        def small_sisters(p):
            r1, r2 = _row(p[4]), _row(p[2])
            if min(r1, r2) < 0 or max(r1, r2) >= len(rad_abs):
                return False
            a, b = float(rad_abs[r1]), float(rad_abs[r2])
            if not (np.isfinite(a) and np.isfinite(b)):
                return False
            return (sz.get(p[4], 9.0) <= cfg.safe_div_sister_size_max and sz.get(p[2], 9.0) <= cfg.safe_div_sister_size_max
                    and abs(a - b) / max((a + b) / 2.0, 1e-6) <= cfg.safe_div_sister_size_sym)
        pair: dict[tuple, float] = {}
        if cfg.div_pair_model and self.frames is not None:
            path = cfg.div_pair_model
            if "{other}" in path:                        # lab LOEO: the pair model trained on the other embryo
                emb = getattr(self, "name", "")[:4]
                path = path.replace("{other}", {"44b6": "6bba", "6bba": "44b6"}.get(emb, "all"))
            dev = cuda_device_for_worker()
            keys, trips = [], []
            for t, (proposals, _) in props_by_t.items():
                for p in proposals:
                    keys.append((t, p[1], p[2], p[4]))
                    trips.append((int(t), zyx(allnodes[p[1]]), zyx(allnodes[p[2]]), zyx(allnodes[p[4]])))
            if trips:
                pr = pair_score_triplets(pair_load(path, dev), self.frames.zarr_path, trips, dev)
                pair = {k: float(v) for k, v in zip(keys, pr)}
        out = {}
        for t, (proposals, cap) in props_by_t.items():
            keep = []
            for p in proposals:
                kind = p[5]
                if cfg.safe_div_size_veto > 0 and sz.get(p[2]) is not None and sz[p[2]] > cfg.safe_div_size_veto:
                    self._count(f"safe_div_rejected_size_{kind}")
                    continue
                a, b, c = st.get(p[1]), st.get(p[2]), st.get(p[4])
                total = -1.0
                if a is None or b is None or c is None:
                    ok = kind == "orphan"
                else:
                    total = a[1] + b[2] + bonus(p[2]) + c[2] + bonus(p[4])
                    if cfg.safe_div_temporal_w > 0:
                        ext = 0.0
                        pr = st.get(thief_of.get(p[1]))
                        if pr is not None:
                            ext += float(pr[1])
                        for i in (succ1.get(p[2]), succ1.get(p[4])):
                            sc_ = st.get(i)
                            if sc_ is not None:
                                ext += float(sc_[2])
                        total += cfg.safe_div_temporal_w * ext
                    ok = total >= need[kind]
                    if kind == "steal" and not ok and cfg.safe_div_steal_parent_min > 0:      # confident mother: lower sum bar
                        ok = a[1] >= cfg.safe_div_steal_parent_min and total >= cfg.safe_div_steal_p_sum_min
                    if not ok and use_align and kind in ("steal", "orphan", "orphan_nd", "orphan_asym", "orphan_traj") and aligned(p):
                        ok = total >= (cfg.safe_div_align_steal_min if kind == "steal" else cfg.safe_div_align_orphan_min)
                        if ok:
                            self._count(f"safe_div_align_{kind}")
                    if not ok and use_size and kind == "steal" and small_sisters(p):
                        ok = total >= cfg.safe_div_sister_size_steal_min
                        if ok:
                            self._count("safe_div_size_steal")
                    if kind == "steal" and p[2] in thief_of:
                        x_id = thief_of[p[2]]
                        sx = st.get(x_id)
                        if ok and cfg.safe_div_steal_thief_parent_max < 1.0 and sx is not None and sx[1] > cfg.safe_div_steal_thief_parent_max:
                            ok = False                                    # the victim is a mother itself: q is probably its daughter
                            self._count("safe_div_rejected_thief_mother")
                        if not ok and cfg.safe_div_steal_alt_min > 0 and total >= cfg.safe_div_steal_alt_min \
                                and (sx is None or sx[1] <= cfg.safe_div_steal_thief_parent_max):
                            q2 = victim_alternative(x_id, p[2], t + 1)
                            if q2 is not None:
                                ok = True
                                self._steal_alt[(p[1], p[2])] = (x_id, q2)
                                self._count("safe_div_steal_alt")
                pp = pair.get((t, p[1], p[2], p[4]))
                if pp is not None and np.isfinite(pp):
                    if cfg.div_pair_mode == "replace":
                        ok = pp >= cfg.div_pair_min
                    elif cfg.div_pair_mode == "or":
                        ok = ok or pp >= cfg.div_pair_or_min
                    else:                                    # "and"
                        ok = ok and pp >= cfg.div_pair_min
                    if not ok:
                        self._count(f"safe_div_rejected_pair_{kind}")
                if ok:
                    keep.append(p)
                else:
                    self._count(f"safe_div_rejected_cnn_{kind}")
            out[t] = (keep, cap)
        return out

    def learned_divisions(self, nodes: dict, edges: list) -> list:
        """Forks from a learned model over candidate triplets (parent, its child, second daughter)."""
        cfg = self.cfg
        if not nodes or not edges or self.cache is None:
            return edges
        dc_fn = ((lambda t, p: self.dc.score(self.frames, t, p)) if (self.dc is not None and self.frames is not None
                                                                         and cfg.deepcenter) else None)
        if cfg.div_mitosis_model and getattr(self.cache, "mit", None) is None and self.frames is not None:
            # cell-state scores of every strong detection and of the weak ones that made it into the graph
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            need = np.zeros(len(self.cache.txyz), dtype=bool)
            need[:self.cache.n_strong] = True
            for n in nodes.values():
                ids, _ = self.cache.near(int(n["t"]), point_um(n), 0.6)
                if len(ids):
                    need[ids[0]] = True
            mit = np.full((len(self.cache.txyz), 3), np.nan, dtype=np.float32)
            mit[need] = mitosis_score_nodes(mitosis_load(cfg.div_mitosis_model, dev), self.frames.zarr_path,
                                            self.cache.txyz[need], dev)
            self.cache.mit = mit
        cand = division_candidates(nodes, edges, self.cache, self.flow, self.frames, cfg, dc_fn)
        if cfg.div_dump_dir:
            save_division_dump(Path(cfg.div_dump_dir) / f"{getattr(self, 'name', 'video')}.npz", nodes, cand)
        if not cfg.div_learn or not cfg.div_model_path or not len(cand["s"]):
            return edges
        path = cfg.div_model_path
        if "{other}" in path:     # leave-one-embryo-out in the lab: the model trained on the other embryo
            emb = getattr(self, "name", "")[:4]
            path = path.replace("{other}", {"44b6": "6bba", "6bba": "44b6"}.get(emb, "all"))
        model = division_model(path)
        cols = [DIV_FEATURES.index(f) for f in model.feature_name()]
        score = model.predict(cand["X"][:, cols])
        parent_edge = {e["target_id"]: k for k, e in enumerate(edges)}
        n_out: dict[int, int] = {}
        for e in edges:
            n_out[e["source_id"]] = n_out.get(e["source_id"], 0) + 1
        used_s, used_q, drop, added = set(), set(), set(), []
        for k in np.argsort(-score):
            if score[k] < cfg.div_threshold:
                break
            s, q = int(cand["s"][k]), int(cand["q"][k])
            if s in used_s or q in used_q or n_out.get(s, 0) != 1:
                continue
            if q < 0:                                    # a dropped detection becomes the second daughter
                p = cand["pseudo"][q]
                nid = max(nodes) + 1
                nodes[nid] = {"node_id": nid, "t": p["t"], "z": p["z"], "y": p["y"], "x": p["x"]}
                used_q.add(q)
                q = nid
                self._count("div_dropped_daughters")
            if q in parent_edge:
                if not cfg.div_steal or score[k] < cfg.div_steal_threshold:
                    continue
                x = edges[parent_edge[q]]["source_id"]
                if x in used_s:
                    continue
                drop.add(parent_edge[q])
                n_out[x] = n_out.get(x, 1) - 1
                self._count("div_steals")
            added.append({"source_id": s, "target_id": q, "edge_prob": None,
                          "distance_um": float(cand["X"][k, DIV_FEATURES.index("d_sq")])})
            used_s.add(s)
            used_q.add(q)
            n_out[s] = n_out.get(s, 0) + 1
        self._count("learned_divisions", len(added))
        return [e for k, e in enumerate(edges) if k not in drop] + added

    # ---- 5b. one lineage cannot fork twice within div_min_age frames ----
    def young_fork_veto(self, nodes: dict, edges: list) -> list:
        """In the 199 training clips (100 frames) no daughter born inside the clip divides again, so a fork whose
        parent track itself starts at a fork less than div_min_age frames earlier is a lineage with one false fork.
        The loser (by CNN parent score, or simply the earlier/later one) loses its repair-added (or longer) daughter
        edge; that daughter then starts a track of its own."""
        cfg = self.cfg
        out_by_src: dict[int, list[int]] = {}
        parent_of: dict[int, int] = {}
        for k, e in enumerate(edges):
            out_by_src.setdefault(e["source_id"], []).append(k)
            parent_of[e["target_id"]] = k
        dropped: set[int] = set()

        def outs(nid):
            return [k for k in out_by_src.get(nid, []) if k not in dropped]

        forks = sorted((nid for nid in out_by_src if len(outs(nid)) == 2), key=lambda i: nodes[i]["t"])
        pp: dict[int, float] = {}
        if cfg.div_min_age_keep == "cnn" and forks:
            for i, s in self._cell_state(nodes, forks).items():
                pp[i] = float(s[1])

        def undo(f):
            ks = outs(f)
            if len(ks) != 2:
                return
            ea, eb = edges[ks[0]], edges[ks[1]]
            if (ea["edge_prob"] is None) != (eb["edge_prob"] is None):       # the edge a repair stage added
                k = ks[0] if ea["edge_prob"] is None else ks[1]
            else:                                                             # else the longer one
                k = ks[0] if float(ea.get("distance_um") or 0.0) >= float(eb.get("distance_um") or 0.0) else ks[1]
            dropped.add(k)
            self._count("young_fork_vetoes")

        for f in forks:
            if len(outs(f)) != 2:
                continue
            t_f = int(nodes[f]["t"])
            cur, age, birth = f, 0, None
            while age < cfg.div_min_age:
                k = parent_of.get(cur)
                if k is None or k in dropped:
                    break
                prev = edges[k]["source_id"]
                age = t_f - int(nodes[prev]["t"])
                if len(outs(prev)) == 2:
                    birth = prev
                    break
                cur = prev
            if birth is None or age >= cfg.div_min_age:
                continue
            if cfg.div_min_age_keep == "first":
                loser = f
            elif cfg.div_min_age_keep == "last":
                loser = birth
            else:
                loser = f if pp.get(f, 0.5) <= pp.get(birth, 0.5) else birth
            undo(loser)
        return [e for k, e in enumerate(edges) if k not in dropped]

    # ---- 5c. divisions are tangential to the tissue surface: undo near-radial forks ----
    def fork_radial_veto(self, nodes: dict, edges: list) -> list:
        """From gastrulation on, EVL and epiblast divisions lie in the plane of the epithelium (research/mitosis_cues_2.md);
        on our GT 2 % of divisions have a sister vector within 30 deg of the local surface normal (0 % of the true forks
        we find) against 7 % of the unmatched forks.  The normal is the smallest principal axis of the strong detections
        within fork_radial_radius_um of the parent in the daughters' frame; a fork closer than fork_radial_deg to it
        loses its repair-added (else longer) daughter edge.  Skipped where the normal is ill-defined (isotropic cloud)."""
        cfg = self.cfg
        if self.cache is None:
            return edges
        out_by_src: dict[int, list[int]] = {}
        for k, e in enumerate(edges):
            out_by_src.setdefault(e["source_id"], []).append(k)
        ns = self.cache.n_strong
        st_t = np.asarray(self.cache.t[:ns]).astype(int)
        st_um = np.asarray(self.cache.um[:ns])
        cos_max = math.cos(math.radians(cfg.fork_radial_deg))
        dropped: set[int] = set()
        for p, ks in out_by_src.items():
            if len(ks) != 2 or p not in nodes:
                continue
            c1, c2 = nodes.get(edges[ks[0]]["target_id"]), nodes.get(edges[ks[1]]["target_id"])
            if c1 is None or c2 is None:
                continue
            P = point_um(nodes[p])
            pts = st_um[st_t == int(c1["t"])]
            if len(pts) < 12:
                continue
            near = pts[np.linalg.norm(pts - P, axis=1) <= cfg.fork_radial_radius_um]
            if len(near) < 12:
                continue
            c = near - near.mean(0)
            w, v = np.linalg.eigh(c.T @ c / len(near))
            aniso = math.sqrt(max(w[0], 1e-9) / max(w[1], 1e-9))
            if aniso > cfg.fork_radial_max_aniso:
                continue
            s = point_um(c1) - point_um(c2)
            n = float(np.linalg.norm(s))
            if n < 1e-6 or abs(float(s @ v[:, 0])) / n < cos_max:
                continue
            ea, eb = edges[ks[0]], edges[ks[1]]
            if (ea["edge_prob"] is None) != (eb["edge_prob"] is None):
                k = ks[0] if ea["edge_prob"] is None else ks[1]
            else:
                k = ks[0] if float(ea.get("distance_um") or 0.0) >= float(eb.get("distance_um") or 0.0) else ks[1]
            dropped.add(k)
            self._count("fork_radial_vetoes")
        return [e for k, e in enumerate(edges) if k not in dropped]

    # ---- 6. isolated nodes and short components ----
    @staticmethod
    def prune_isolated(nodes: dict, edges: list) -> tuple[dict, list]:
        incident = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
        if not incident:
            return nodes, edges
        nodes = {i: n for i, n in nodes.items() if i in incident}
        return nodes, [e for e in edges if e["source_id"] in nodes and e["target_id"] in nodes]

    def prune_weak_leaves(self, nodes: dict, edges: list) -> tuple[dict, list]:
        """Public v4 rule: drop a terminal node (no successor, a single parent that is not a fork, not on the last frame)
        whose only edge carries a learned probability below leaf_prune_min_prob. Single pass, no cascade."""
        cfg = self.cfg
        if cfg.leaf_prune_min_prob <= 0 or not edges or not nodes:
            return nodes, edges
        t_max = max(int(n["t"]) for n in nodes.values())
        out_deg: dict[int, int] = {}
        incoming: dict[int, list[dict]] = {}
        for e in edges:
            out_deg[e["source_id"]] = out_deg.get(e["source_id"], 0) + 1
            incoming.setdefault(e["target_id"], []).append(e)
        drop: set[int] = set()
        for nid, es in incoming.items():
            n = nodes.get(nid)
            if n is None or len(es) != 1 or out_deg.get(nid, 0) or int(n["t"]) >= t_max:
                continue
            if out_deg.get(es[0]["source_id"], 0) >= 2:          # a fork daughter is never pruned here
                continue
            p = es[0].get("edge_prob")
            if p is not None and np.isfinite(float(p)) and float(p) < cfg.leaf_prune_min_prob:
                drop.add(nid)
        if not drop:
            return nodes, edges
        self._count("leaf_pruned_nodes", len(drop))
        nodes = {i: n for i, n in nodes.items() if i not in drop}
        return nodes, [e for e in edges if e["target_id"] not in drop]

    def filter_short(self, nodes: dict, edges: list) -> tuple[dict, list]:
        cfg = self.cfg
        if cfg.min_track_len <= 1 or not edges:
            return nodes, edges
        parent = {i: i for i in nodes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        out_count: dict[int, int] = {}
        for e in edges:
            a, b = e["source_id"], e["target_id"]
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            out_count[a] = out_count.get(a, 0) + 1
        comps: dict[int, list[int]] = {}
        for i in nodes:
            comps.setdefault(find(i), []).append(i)
        keep: set[int] = set()
        t_max = max(int(n["t"]) for n in nodes.values())
        for members in comps.values():
            has_div = any(out_count.get(i, 0) >= 2 for i in members)
            at_edge = cfg.short_keep_boundary and len(members) >= 2 and any(
                int(nodes[i]["t"]) in (0, t_max) for i in members)
            if len(members) >= cfg.min_track_len or (cfg.keep_division_components and has_div) or at_edge:
                keep.update(members)
        if not keep or len(keep) == len(nodes):
            return nodes, edges
        if cfg.short_track_rescue and (len(nodes) - len(keep)) / max(len(nodes), 1) >= cfg.rescue_trigger_removed_frac:
            comp_edges: dict[int, list[dict]] = {}
            for e in edges:
                if e["source_id"] in parent and e["target_id"] in parent:
                    comp_edges.setdefault(find(e["source_id"]), []).append(e)
            budget = min(cfg.rescue_max_nodes_abs, max(0, int(round(len(nodes) * cfg.rescue_max_nodes_frac))))
            proposals = []
            for root, members in comps.items():
                if set(members) & keep or not (cfg.rescue_min_len <= len(members) < cfg.min_track_len):
                    continue
                ce = comp_edges.get(root, [])
                if not ce:
                    continue
                # an edge without a learned probability (gap / division repair) counts as 0, as in the public code
                probs = [float(e["edge_prob"]) if e.get("edge_prob") is not None else 0.0 for e in ce]
                dists = [float(e["distance_um"]) for e in ce if np.isfinite(float(e.get("distance_um", np.nan)))]
                mean_p = float(np.mean(probs)) if probs else 0.0
                mean_d = float(np.mean(dists)) if dists else float("inf")
                if mean_p < cfg.rescue_min_mean_prob or mean_d > cfg.rescue_max_mean_dist_um:
                    continue
                proposals.append((mean_p - 0.02 * mean_d + 0.004 * len(members), len(members), mean_p, root, members))
            proposals.sort(reverse=True)
            used = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or used + size > budget:
                    continue
                keep.update(members)
                used += size
            self._count("rescued_nodes", used)
        nodes = {i: n for i, n in nodes.items() if i in keep}
        return nodes, [e for e in edges if e["source_id"] in nodes and e["target_id"] in nodes]

    # ---- 7. line-fit smoothing of unbranched track interiors (topology unchanged) ----
    def linefit(self, nodes: dict, edges: list) -> dict:
        cfg = self.cfg
        if cfg.linefit_weight <= 0 or cfg.linefit_window <= 0 or not edges:
            return nodes
        pred, succ = {}, {}
        for e in edges:
            s, t = nodes.get(e["source_id"]), nodes.get(e["target_id"])
            if s is None or t is None or t["t"] != s["t"] + 1:
                continue
            succ.setdefault(e["source_id"], []).append(e["target_id"])
            pred.setdefault(e["target_id"], []).append(e["source_id"])
        orig = {i: np.array(zyx(n), dtype=np.float64) for i, n in nodes.items()}
        w = float(np.clip(cfg.linefit_weight, 0.0, 1.0))
        updates = {}
        for nid in sorted(nodes):
            hood = [(0, nid)]
            for links, sign in ((pred, -1), (succ, 1)):
                cur = nid
                for step in range(1, cfg.linefit_window + 1):
                    nxt = links.get(cur, [])
                    if len(nxt) != 1 or nxt[0] not in orig:
                        break
                    cur = nxt[0]
                    hood.append((sign * step, cur))
            if len(hood) < 3:
                continue
            dts = np.array([d for d, _ in hood], dtype=np.float64)
            xyz = np.stack([orig[i] for _, i in hood])
            fit = np.array([np.polyval(np.polyfit(dts, xyz[:, a], 1), 0.0) for a in range(3)])
            if np.isfinite(fit).all():
                updates[nid] = (1.0 - w) * orig[nid] + w * fit
        for nid, p in updates.items():
            nodes[nid]["z"], nodes[nid]["y"], nodes[nid]["x"] = float(p[0]), float(p[1]), float(p[2])
        return nodes

    def run(self, nodes: dict, raw_edges: list) -> tuple[dict, list]:
        cfg = self.cfg
        edges = self.clean_edges(nodes, raw_edges)
        if cfg.relink_flow or cfg.gap_flow or cfg.safe_div_flow or cfg.rescue_weak or cfg.stitch:
            self.flow = Flow(nodes, edges, cfg)          # from the ILP links, before any repair touches them
            self._count("flow_links", self.flow.n_links)
        if cfg.center_stage == "pre" and (cfg.center_model or cfg.center_offset_dir) and self.cache is not None:
            nodes = self.apply_center_offsets(nodes)
        if cfg.relink:
            edges = self.motion_relink(nodes, edges)
        edges = self.single_parent(edges)
        nodes, edges = self.rescue_weak(nodes, edges)
        nodes, edges = self.close_gaps(nodes, edges)
        nodes, edges = self.close_gap2(nodes, edges)
        nodes, edges = self.stitch_tracks(nodes, edges)
        if cfg.div_learn or cfg.div_dump_dir:
            edges = self.learned_divisions(nodes, edges)
        edges = self.safe_divisions(nodes, edges)
        if cfg.div_min_age > 0:
            edges = self.young_fork_veto(nodes, edges)
        if cfg.fork_radial_deg > 0:
            edges = self.fork_radial_veto(nodes, edges)
        if cfg.prune_isolated:
            nodes, edges = self.prune_isolated(nodes, edges)
        nodes, edges = self.prune_weak_leaves(nodes, edges)
        nodes, edges = self.filter_short(nodes, edges)
        if cfg.refine_all:
            nodes = self.refine_all_nodes(nodes)
        nodes = self.linefit(nodes, edges)
        if cfg.center_stage != "pre" and (cfg.center_model or cfg.center_offset_dir) and self.cache is not None:
            nodes = self.apply_center_offsets(nodes)
        if any(float(v) != 0.0 for v in cfg.coord_shift_zyx):
            dz, dy, dx = (float(v) for v in cfg.coord_shift_zyx)
            for n in nodes.values():
                n["z"], n["y"], n["x"] = float(n["z"]) + dz, float(n["y"]) + dy, float(n["x"]) + dx
        return nodes, edges

    def apply_center_offsets(self, nodes: dict) -> dict:
        """Move every node that is a cached strong detection by its learned click-offset vector (voxels)."""
        cfg = self.cfg
        off = getattr(self.cache, "center_off", None)
        if off is None and cfg.center_model and self.frames is not None and self.cache is not None:
            import torch
            dev = cuda_device_for_worker()
            ns = self.cache.n_strong
            off = np.zeros((len(self.cache.txyz), 3), dtype=np.float32)
            off[:ns] = center_score_nodes(center_load(cfg.center_model, dev), self.frames.zarr_path,
                                          self.cache.txyz[:ns], dev, tta=cfg.center_tta)
            torch.cuda.empty_cache()
            self.cache.center_off = off
        if off is None:
            return nodes
        ns = self.cache.n_strong
        moved = 0
        for nid, n in nodes.items():
            ci = None
            if nid < ns and self.cache.t[nid] == n["t"]:
                ci = nid
            else:
                ids, _ = self.cache.near(int(n["t"]), point_um(n), 2.5, "strong")
                if len(ids):
                    ci = int(ids[0])
            if ci is None or not np.all(np.isfinite(off[ci])):
                continue
            n["z"], n["y"], n["x"] = (float(n["z"]) + cfg.center_scale * float(off[ci][0]),
                                      float(n["y"]) + cfg.center_scale * float(off[ci][1]),
                                      float(n["x"]) + cfg.center_scale * float(off[ci][2]))
            moved += 1
        self.stats["center_moved"] = moved
        return nodes

    def refine_all_nodes(self, nodes: dict) -> dict:
        """Intensity-centroid refinement (same window and shift limit as for synthetic nodes) of every node."""
        if self.frames is None:
            return nodes
        by_t: dict[int, list[int]] = {}
        for nid, n in nodes.items():
            by_t.setdefault(int(n["t"]), []).append(nid)
        moved = 0
        for t in sorted(by_t):
            for nid in by_t[t]:
                n = nodes[nid]
                p = (float(n["z"]), float(n["y"]), float(n["x"]))
                q = self.refine(t, p)
                if q != p:
                    n["z"], n["y"], n["x"] = q
                    moved += 1
        self.stats["refine_all_moved"] = moved
        return nodes


# =============================================================================================
# Output
# =============================================================================================

def graph_rows(dataset: str, nodes: dict, edges: list) -> list[list]:
    rows = []
    for nid in sorted(nodes):
        n = nodes[nid]
        rows.append([dataset, "node", int(nid), int(n["t"]), max(0, int(round(float(n["z"])))),
                     max(0, int(round(float(n["y"])))), max(0, int(round(float(n["x"])))), -1, -1])
    for e in edges:
        rows.append([dataset, "edge", -1, -1, -1, -1, -1, int(e["source_id"]), int(e["target_id"])])
    return rows


def public_repair_config(cfg: "Config") -> "Config":
    """The same config with every repair addition switched off (the public repair behaviour)."""
    return replace(cfg, relink_keep_forks=False, relink_flow=False, gap_flow=False, gap_use_weak=False,
                   rescue_weak=False, rescue_dropped=False, safe_div_flow=False, frozen_frames=False, stitch=False,
                   refine_all=False, short_keep_boundary=False, coord_shift_zyx=[0.0, 0.0, 0.0])


def check_graph(dataset: str, nodes: dict, edges: list) -> None:
    indeg, outdeg = {}, {}
    for e in edges:
        s, t = nodes.get(e["source_id"]), nodes.get(e["target_id"])
        if s is None or t is None:
            raise AssertionError(f"{dataset}: dangling edge {e}")
        if t["t"] != s["t"] + 1:
            raise AssertionError(f"{dataset}: non-consecutive edge {e}")
        indeg[e["target_id"]] = indeg.get(e["target_id"], 0) + 1
        outdeg[e["source_id"]] = outdeg.get(e["source_id"], 0) + 1
    if indeg and max(indeg.values()) > 1:
        raise AssertionError(f"{dataset}: node with two parents")
    if outdeg and max(outdeg.values()) > 2:
        raise AssertionError(f"{dataset}: node with more than two children")


def save_raw(raw_dir: Path, dataset: str, nodes: dict, raw_edges: list) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(nodes), dtype=np.int64)
    xyz = np.array([[nodes[i]["t"], nodes[i]["z"], nodes[i]["y"], nodes[i]["x"]] for i in ids], dtype=np.float32)
    e = np.array([[x["source_id"], x["target_id"]] for x in raw_edges], dtype=np.int64).reshape(-1, 2)
    p = np.array([np.nan if x["edge_prob"] is None else x["edge_prob"] for x in raw_edges], dtype=np.float32)
    np.savez_compressed(raw_dir / f"{dataset}.npz", ids=ids, txyz=xyz, edges=e, probs=p)


def save_cache(cache_dir: Path, dataset: str, cache: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{dataset}.npz", **cache)


def load_raw(path: Path) -> tuple[dict, list]:
    d = np.load(path)
    nodes = {int(i): {"node_id": int(i), "t": int(r[0]), "z": float(r[1]), "y": float(r[2]), "x": float(r[3])}
             for i, r in zip(d["ids"], d["txyz"])}
    edges = [{"source_id": int(a), "target_id": int(b), "edge_prob": None if np.isnan(p) else float(p)}
             for (a, b), p in zip(d["edges"], d["probs"])]
    return nodes, edges


def list_datasets(data_dir: Path) -> list[str]:
    return sorted(p.name[:-5] for p in Path(data_dir).iterdir() if p.name.endswith(".zarr"))


# =============================================================================================
# Train duplicates: recognise a test video that is a copy of a train recording and reuse its annotation
# =============================================================================================

SIG_SHAPE = (8, 16, 16)


def frame_signature(vol: np.ndarray) -> np.ndarray:
    """(Z, Y, X) -> z-scored (8, 16, 16) float16 by block mean pooling (adaptive to the frame shape)."""
    Z, Y, X = vol.shape
    zs, ys, xs = SIG_SHAPE
    v = vol.astype(np.float32)
    v = v[: (Z // zs) * zs, : (Y // ys) * ys, : (X // xs) * xs]
    s = v.reshape(zs, Z // zs, ys, Y // ys, xs, X // xs).mean(axis=(1, 3, 5))
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s.astype(np.float16)


def _unit_rows(sig: np.ndarray) -> np.ndarray:
    s = sig.astype(np.float32).reshape(len(sig), -1)
    s = s - s.mean(axis=1, keepdims=True)
    return s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)


class TrainSigs:
    """Signatures of every train frame (train_sigs.py): names, frame index, unit-norm rows for NCC by dot product."""

    def __init__(self, path: str):
        with np.load(path) as d:
            self.names = d["names"].astype(str)
            self.t = d["t"].astype(np.int64)
            self.S = _unit_rows(d["sig"])
        self.videos = sorted(set(self.names.tolist()))

    def frames_of(self, video: str) -> dict[int, int]:
        idx = np.nonzero(self.names == video)[0]
        return {int(self.t[i]): int(i) for i in idx}


def _ncc_full(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_train_duplicate(zarr_path: Path, sigs: TrainSigs, train_dir: Path, cfg: "Config") -> dict | None:
    """The train video (and time offset) this video is a copy of, or None. Two independent checks: every frame's coarse
    signature must match the train frame at one common offset (t_train = t + offset), and full-resolution frames spread
    over the video must agree too. Ambiguity between videos is impossible at the threshold (cross-video NCC <= 0.965)."""
    shape, _ = zarr_meta(zarr_path)
    T = int(shape[0])
    sig = _unit_rows(np.stack([frame_signature(read_frame(zarr_path, t)) for t in range(T)]))
    ncc = sig @ sigs.S.T                                           # (T, train frames)
    best = ncc.argmax(axis=1)
    hit = ncc[np.arange(T), best] >= cfg.dup_ncc_min
    if hit.sum() < max(1, cfg.dup_min_cover * T):
        return None
    videos, counts = np.unique(sigs.names[best[hit]], return_counts=True)
    video = str(videos[counts.argmax()])
    frames = sigs.frames_of(video)
    sel = hit & (sigs.names[best] == video)
    offsets = np.unique(sigs.t[best[sel]] - np.arange(T)[sel])
    best_fit = None
    for d in offsets:
        rows = [(t, frames[t + int(d)]) for t in range(T) if (t + int(d)) in frames]
        if len(rows) < cfg.dup_min_cover * T:
            continue
        vals = ncc[[r[0] for r in rows], [r[1] for r in rows]]
        fit = (float(vals.mean()), float(vals.min()), int(d), len(rows))
        if best_fit is None or fit > best_fit:
            best_fit = fit
    if best_fit is None or best_fit[1] < cfg.dup_ncc_min:
        return None
    mean_ncc, min_ncc, d, n_cov = best_fit
    train_zarr = train_dir / f"{video}.zarr"
    if not train_zarr.exists() or not (train_dir / f"{video}.geff").exists():
        return None
    tshape, _ = zarr_meta(train_zarr)
    if tuple(tshape[1:]) != tuple(shape[1:]):
        return None
    covered = [t for t in range(T) if (t + d) in frames]
    checks = sorted({covered[int(round(q * (len(covered) - 1)))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)})
    full, exact = [], True
    for t in checks:
        a, b = read_frame(zarr_path, t), read_frame(train_zarr, t + d)
        full.append(_ncc_full(a, b))
        exact = exact and a.shape == b.shape and bool(np.array_equal(a, b))
    if min(full) < cfg.dup_full_ncc_min:
        return None
    return {"video": video, "offset": int(d), "cover": n_cov / T, "sig_ncc_min": round(min_ncc, 5),
            "sig_ncc_mean": round(mean_ncc, 5), "full_ncc_min": round(min(full), 5), "exact": int(exact)}


def gt_graph_from_geff(geff_path: Path, offset: int, n_frames: int) -> tuple[dict, list]:
    """The train annotation as (nodes, edges) in this pipeline's format, shifted by -offset and cut to [0, n_frames).
    Edges that the submission format cannot carry (non-consecutive frames, a third child) are dropped."""
    import zarr
    g = zarr.open(str(geff_path), mode="r")
    ids = np.asarray(g["nodes"]["ids"][:]).astype(np.int64)
    props = {k: np.asarray(g["nodes"]["props"][k]["values"][:]) for k in ("t", "z", "y", "x")}
    t = props["t"].astype(np.int64) - int(offset)
    keep = (t >= 0) & (t < n_frames)
    remap = {int(i): k for k, i in enumerate(ids[keep])}
    nodes = {}
    for k, i in enumerate(np.nonzero(keep)[0]):
        nodes[k] = {"node_id": k, "t": int(t[i]), "z": float(props["z"][i]), "y": float(props["y"][i]),
                    "x": float(props["x"][i])}
    edges, n_children = [], {}
    e_arr = g["edges"]["ids"]
    e_ids = np.asarray(e_arr[:]).astype(np.int64).reshape(-1, 2) if e_arr.shape[0] else np.zeros((0, 2), np.int64)
    for s, u in e_ids:
        a, b = remap.get(int(s)), remap.get(int(u))
        if a is None or b is None or nodes[b]["t"] != nodes[a]["t"] + 1 or n_children.get(a, 0) >= 2:
            continue
        n_children[a] = n_children.get(a, 0) + 1
        edges.append({"source_id": a, "target_id": b, "prob": 1.0})
    return nodes, edges


# =============================================================================================
# CLI
# =============================================================================================

def threshold_candidates(cache: "RichCache", cfg: "Config", src, tgt, prob) -> list:
    keep = prob > cfg.edge_threshold
    src, tgt, prob = src[keep], tgt[keep], prob[keep]
    dist = np.linalg.norm(cache.um[src] - cache.um[tgt], axis=1)
    if cfg.edge_max_um > 0:
        ok = dist <= cfg.edge_max_um
        src, tgt, prob, dist = src[ok], tgt[ok], prob[ok], dist[ok]
    return [(int(a), int(b), float(p), float(d)) for a, b, p, d in zip(src, tgt, prob, dist)]


def link_video(coords, cand, cache, models, cfg: "Config", info: dict, frames=None, mit=None, node_cost=None):
    """Global linking of one video: optional learned re-scoring or motion prior, the ILP, the adaptive track-end cost."""
    if cfg.ilp_off:
        return no_ilp_graph(coords, cand)
    if cfg.ilp_div_track_max > 0 and cfg.ilp_div_cnn and mit is not None and len(mit) == len(coords):
        # the mother's "parent" evidence may peak a frame or two before the fork: carry the max P(parent) along the
        # predecessors of a first-pass plain ILP (as the lab does with the cached baseline graph) into the division cost
        first = [c for c in cand if c[2] > cfg.ilp_track_first_thr]
        _, e0 = solve_ilp(coords, first, models, replace(cfg, ilp_div_cnn=False))
        pred = {int(e["target_id"]): int(e["source_id"]) for e in e0}
        mp = np.nan_to_num(np.asarray(mit[:, 1], dtype=np.float64), nan=0.0)
        mp_track = mp.copy()
        for i in range(len(mp)):
            cur, best = i, mp[i]
            for _ in range(cfg.ilp_div_track_max):
                cur = pred.get(cur)
                if cur is None or cur >= len(mp):
                    break
                best = max(best, mp[cur])
            mp_track[i] = best
        mit = np.array(mit, dtype=np.float32, copy=True)
        mit[:, 1] = mp_track
        info["ilp_track_max_raised"] = int((mp_track > mp + 1e-9).sum())
    if cfg.rescore_model_path and cache is not None:
        rc = RichCache(cache)
        fn, fe = candidate_flow_graph(coords, cand, cfg)
        flow = Flow(fn, fe, cfg)
        src, tgt, prob = rescored_probs(rc, cfg, flow, frames, cfg.rescore_model_path, cfg.rescore_device,
                                        drop=tuple(cfg.rescore_drop))
        if cfg.ilp_flow_penalty > 0:
            disp = rc.um[tgt] - rc.um[src]
            usrc, inv = np.unique(src, return_inverse=True)
            upred = np.stack([flow.at(int(rc.t[i]), rc.um[i], own=rc.um[i]) for i in usrc])
            resid = np.linalg.norm(disp - upred[inv], axis=1)
            z = np.log(np.clip(prob, 1e-12, 1)) - cfg.ilp_flow_penalty * (resid / cfg.ilp_flow_sigma_um) ** 2 / 2.0
            m = np.full(len(rc.txyz), -np.inf)
            np.maximum.at(m, tgt, z)
            ssum = np.zeros(len(rc.txyz))
            np.add.at(ssum, tgt, np.exp(z - m[tgt]))
            prob = np.exp(z - m[tgt]) / ssum[tgt]
        cand = threshold_candidates(rc, cfg, src, tgt, prob)
        info["rescored_candidates"] = len(cand)
    elif cfg.ilp_flow_penalty > 0 and cache is not None and cfg.ilp_flow_source == "candidates":
        # motion prior from confident mutual-best candidates: a single ILP, as in the baseline
        fn, fe = candidate_flow_graph(coords, cand, cfg)
        cand = flow_penalized_candidates(RichCache(cache), cfg, fn, fe)
        info["ilp_flow_candidates"] = len(cand)
    nodes, raw_edges = solve_ilp(coords, cand, models, cfg, mit, node_cost)
    if cfg.ilp_flow_penalty > 0 and cache is not None and cfg.ilp_flow_source == "ilp" and not cfg.rescore_model_path:
        # motion prior inside the global solve: second ILP on flow-penalised link probabilities
        cand = flow_penalized_candidates(RichCache(cache), cfg, nodes, raw_edges)
        nodes, raw_edges = solve_ilp(coords, cand, models, cfg, mit)
        info["ilp_flow_candidates"] = len(cand)
    cost, unc = adaptive_ilp_cost(raw_edges, cfg)
    if unc == unc:
        info["edge_uncertainty"] = round(unc, 4)
    if cost is not None:
        nodes, raw_edges = (no_ilp_graph(coords, cand) if cost < 0 else
                            solve_ilp(coords, cand, models, replace(cfg, ilp_disappearance=cost), mit, node_cost))
        info["adaptive_ilp_cost"] = cost
    return nodes, raw_edges


def cmd_run(args) -> None:
    import torch
    cfg = Config.load(args.config, **(json.loads(args.override) if args.override else {}))
    data_dir = Path(args.data_dir)
    names = args.videos.split(",") if args.videos else list_datasets(data_dir)
    names = names[args.shard::args.nshards]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    models = None if args.pp_only else TrackingModels(cfg, device)
    dc = DeepCenter(cfg, device) if cfg.deepcenter else None
    print(f"[shard {args.shard}/{args.nshards}] {len(names)} videos | device {device} | "
          f"setup {time.time() - t0:.1f}s | config tta={cfg.tta}", flush=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    degraded = False
    dup_sigs, dup_train_dir = None, None
    if cfg.dup_sigs_path and not args.pp_only:
        dup_train_dir = Path(cfg.dup_train_dir) if cfg.dup_train_dir else data_dir.parent / "train"
        try:
            dup_sigs = TrainSigs(cfg.dup_sigs_path)
            print(f"[shard {args.shard}] train duplicates: {len(dup_sigs.videos)} train videos, {len(dup_sigs.names)} frame "
                  f"signatures; train dir {dup_train_dir} ({'found' if dup_train_dir.is_dir() else 'MISSING'})", flush=True)
            if not dup_train_dir.is_dir():
                dup_sigs = None
        except Exception as exc:
            print(f"[shard {args.shard}] WARNING train duplicates disabled ({exc})", flush=True)
            dup_sigs = None
    n_dup, t_dup = 0, 0.0
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ROW_COLUMNS)
        for k, name in enumerate(names):
            tv = time.time()
            zarr_path = data_dir / f"{name}.zarr"
            if dup_sigs is not None:
                # A copy of a train recording: write its annotation (shifted to this clip's frames) instead of inferring
                match = None
                try:
                    match = find_train_duplicate(zarr_path, dup_sigs, dup_train_dir, cfg)
                    if match is not None:
                        n_frames = int(zarr_meta(zarr_path)[0][0])
                        nodes, edges = gt_graph_from_geff(dup_train_dir / f"{match['video']}.geff", match["offset"], n_frames)
                        if not nodes:
                            raise ValueError("empty annotation")
                        check_graph(name, nodes, edges)
                except Exception as exc:
                    print(f"[shard {args.shard}] WARNING {name}: duplicate check failed ({exc}); inferring", flush=True)
                    match = None
                if match is not None:
                    writer.writerows(graph_rows(name, nodes, edges))
                    fh.flush()
                    n_div = sum(1 for v in np.unique([e["source_id"] for e in edges], return_counts=True)[1] if v >= 2) if edges else 0
                    stats_rows.append({"dataset": name, "raw_nodes": len(nodes), "nodes": len(nodes), "edges": len(edges),
                                       "divisions": int(n_div), "tta": "dup", "predict_s": round(time.time() - tv, 1),
                                       "repair_s": 0.0, **{f"dup_{kk}": v for kk, v in match.items()}})
                    n_dup += 1
                    t_dup += time.time() - tv
                    print(f"[shard {args.shard}] {k + 1}/{len(names)} {name}: DUPLICATE of train {match['video']} "
                          f"(offset {match['offset']}, cover {match['cover']:.2f}, sig ncc >= {match['sig_ncc_min']}, "
                          f"full ncc >= {match['full_ncc_min']}, exact {match['exact']}): {len(nodes)} nodes, {len(edges)} edges "
                          f"from the train annotation in {time.time() - tv:.1f}s", flush=True)
                    continue
            # Time guard: if the projected finish exceeds the budget, finish the remaining videos without TTA
            # (single view, both models kept). Deterministic and logged; never triggers on normal runs.
            if args.time_budget_h and k > n_dup and not degraded and not args.pp_only:
                per_video = (tv - t0 - t_dup) / (k - n_dup)
                if (tv - t0) + per_video * (len(names) - k) > args.time_budget_h * 3600:
                    cfg.tta, degraded = "none", True
                    print(f"[shard {args.shard}] TIME GUARD: switching to single-view inference for the last "
                          f"{len(names) - k} videos", flush=True)
            if args.pp_only:
                nodes, raw_edges = load_raw(Path(args.raw_dir) / f"{name}.npz")
                info = {}
            else:
                need_cache = (cfg.rescue_weak or cfg.gap_use_weak or cfg.frozen_frames      # repair fixes read it
                              or cfg.ilp_flow_penalty > 0 or cfg.div_learn or bool(cfg.rescore_model_path)
                              or cfg.safe_div_dropped_min > 0 or cfg.safe_div_weak_min > 0 or bool(cfg.center_model))
                cache = {} if (args.cache_dir or need_cache) else None
                try:
                    coords, cand, info = predict_video(zarr_path, models, cfg, cache)
                except Exception as exc:
                    # Safety net: retry the detection without TTA and without the cache (e.g. out of memory)
                    print(f"[shard {args.shard}] WARNING {name}: inference failed ({exc}); single view, no cache",
                          flush=True)
                    torch.cuda.empty_cache()
                    cache = None
                    try:
                        coords, cand, info = predict_video(zarr_path, models, replace(cfg, tta="none"), None)
                        info["predict_fallback"] = 1
                    except Exception as exc2:
                        # Last resort: one node, so that the submission still covers this video
                        print(f"[shard {args.shard}] ERROR {name}: inference failed twice ({exc2}); placeholder node",
                              flush=True)
                        shape = zarr_meta(zarr_path)[0]
                        coords = np.array([[0, shape[1] // 2, shape[2] // 2, shape[3] // 2]], dtype=np.int16)
                        cand, info = [], {"predict_fallback": 2}
                cand0 = list(cand)
                mit = None
                try:
                    if not cand0:
                        raise ValueError("no candidate links")
                    if cfg.ilp_div_cnn and cfg.div_mitosis_model:
                        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        t_mit = time.time()
                        mit = mitosis_score_nodes(mitosis_load(cfg.ilp_cnn_model or cfg.div_mitosis_model, dev), zarr_path,
                                                  coords, dev, views=cfg.ilp_cnn_views, norm=cfg.mitosis_norm)
                        if cfg.mit_calib:                        # embryo-shift calibration of the whole video's scores
                            mit = calibrate_mit(mit, len(mit), cfg.mit_calib, cfg.mit_calib_ref)
                        info["ilp_cnn_s"] = round(time.time() - t_mit, 1)
                    node_cost = None
                    if cfg.ilp_dc_weight > 0 and dc is not None:
                        # DeepCenter as a node prior: a detection with a weak centre likelihood pays to be on a track
                        t_dc = time.time()
                        dc.reset()
                        fc_dc = FrameCache(zarr_path)
                        dcs = np.full(len(coords), np.nan)
                        for t in np.unique(coords[:, 0]):
                            for i in np.nonzero(coords[:, 0] == t)[0]:
                                sc = dc.score(fc_dc, int(t), (float(coords[i, 1]), float(coords[i, 2]), float(coords[i, 3])))
                                if sc is not None:
                                    dcs[i] = sc
                        dcs = np.nan_to_num(dcs.astype(np.float16).astype(np.float64), nan=float(cfg.ilp_dc_tau))
                        node_cost = cfg.ilp_dc_weight * np.maximum(cfg.ilp_dc_tau - dcs, 0.0)
                        dc.reset()
                        info["ilp_dc_s"] = round(time.time() - t_dc, 1)
                        info["ilp_dc_nodes_charged"] = int((node_cost > 0).sum())
                    nodes, raw_edges = link_video(coords, cand, cache, models, cfg, info,
                                                  FrameCache(zarr_path) if cfg.rescore_model_path else None, mit, node_cost)
                except Exception as exc:
                    # Safety net: the plain baseline ILP on the original candidates (or bare detections)
                    print(f"[shard {args.shard}] WARNING {name}: linking failed ({exc}); plain ILP", flush=True)
                    try:
                        nodes, raw_edges = solve_ilp(coords, cand0, models, cfg)
                    except Exception:
                        nodes = {i: {"node_id": i, "t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3])}
                                 for i, c in enumerate(coords)}
                        raw_edges = []
                    info["ilp_fallback"] = 1
                if args.raw_dir:
                    save_raw(Path(args.raw_dir), name, nodes, raw_edges)
                if cache is not None and args.cache_dir:
                    # ILP node ids must index the detection array, so that the cache and the raw graph agree
                    bad = sum(1 for i, n in nodes.items()
                              if i >= len(coords) or (int(n["t"]), int(n["z"])) != (int(coords[i, 0]), int(coords[i, 1])))
                    if bad:
                        print(f"[shard {args.shard}] WARNING {name}: {bad} ILP node ids do not match the detection order",
                              flush=True)
                    info.update({"n_weak": int(len(cache["node_txyz"]) - int(cache["n_strong"])),
                                 "n_ext_pairs": int(len(cache["ext_src"])), "n_base_pairs": int(len(cache["base_src"])),
                                 "frozen_transitions": int(cache["frozen"].sum()), "id_mismatch": int(bad)})
                    save_cache(Path(args.cache_dir), name, cache)
            t_pred = time.time() - tv
            if dc is not None:
                dc.reset()
            frame_cache = FrameCache(zarr_path)
            if not args.pp_only and cache is not None and dc is not None and args.cache_dir:
                # DeepCenter score of every cached node, so offline experiments can run the division step without the model
                txyz = cache["node_txyz"]
                dc_scores = np.full(len(txyz), np.nan, dtype=np.float32)
                for t in np.unique(txyz[:, 0]):
                    for i in np.nonzero(txyz[:, 0] == t)[0]:
                        s = dc.score(frame_cache, int(t), (float(txyz[i, 1]), float(txyz[i, 2]), float(txyz[i, 3])))
                        if s is not None:
                            dc_scores[i] = s
                cache["dc_score"] = dc_scores.astype(np.float16)
                save_cache(Path(args.cache_dir), name, cache)   # overwrite with the DeepCenter column added
                info["dc_score_s"] = round(time.time() - tv - t_pred, 1)
            rich = RichCache(cache) if (not args.pp_only and cache is not None) else None
            if rich is not None and mit is not None and cfg.mit_calib and len(mit) <= len(rich.txyz):
                # the division gates must judge with the same calibrated scores as the ILP (strong rows first in the cache)
                full = np.full((len(rich.txyz), 3), np.nan, dtype=np.float32)
                full[:len(mit)] = mit
                rich.mit = full
            repair = Repair(cfg, frame_cache, dc, rich)
            n_raw = len(nodes)
            nodes_in, edges_in = {k: dict(v) for k, v in nodes.items()}, [dict(e) for e in raw_edges]
            try:
                nodes, edges = repair.run(nodes, raw_edges)
                if not nodes:
                    raise AssertionError(f"{name}: repair removed every node")
                check_graph(name, nodes, edges)
            except Exception as exc:
                # Safety net: one bad video must never sink a whole submission. Redo it with the public repair.
                print(f"[shard {args.shard}] WARNING {name}: repair failed ({exc}); using the public repair", flush=True)
                if dc is not None:
                    dc.reset()
                try:
                    repair = Repair(public_repair_config(cfg), frame_cache, dc, None)
                    nodes, edges = repair.run(nodes_in, edges_in)
                    check_graph(name, nodes, edges)
                    info["repair_fallback"] = 1
                except Exception as exc2:
                    print(f"[shard {args.shard}] WARNING {name}: public repair failed too ({exc2}); raw ILP graph",
                          flush=True)
                    nodes, edges = nodes_in, [e for e in edges_in if e["source_id"] in nodes_in and e["target_id"] in nodes_in]
                    info["repair_fallback"] = 2
            writer.writerows(graph_rows(name, nodes, edges))
            fh.flush()
            n_div = sum(1 for v in np.unique([e["source_id"] for e in edges], return_counts=True)[1] if v >= 2) if edges else 0
            stats_rows.append({"dataset": name, "raw_nodes": n_raw, "nodes": len(nodes), "edges": len(edges),
                               "divisions": int(n_div), "tta": cfg.tta, "predict_s": round(t_pred, 1),
                               "repair_s": round(time.time() - tv - t_pred, 1), **info, **repair.stats})
            print(f"[shard {args.shard}] {k + 1}/{len(names)} {name}: nodes {n_raw}->{len(nodes)} edges {len(edges)} "
                  f"div {n_div} | predict {t_pred:.0f}s repair {time.time() - tv - t_pred:.0f}s | "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    Path(str(out) + ".stats.json").write_text(json.dumps(stats_rows, indent=1))


def cmd_merge(args) -> None:
    expected = set(list_datasets(Path(args.data_dir)))
    seen: set[str] = set()
    row_id = 0
    with open(args.out, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["id", *ROW_COLUMNS])
        for part in args.parts:
            with open(part, newline="") as fi:
                reader = csv.reader(fi)
                header = next(reader)
                if header != ROW_COLUMNS:
                    raise ValueError(f"{part}: bad header {header}")
                for row in reader:
                    seen.add(row[0])
                    writer.writerow([row_id, *row])
                    row_id += 1
    missing, extra = sorted(expected - seen), sorted(seen - expected)
    if missing or extra:
        raise AssertionError(f"submission coverage: missing={missing[:10]} extra={extra[:10]}")
    print(f"wrote {args.out}: {row_id} rows, {len(seen)} datasets")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default=None, help="JSON file with Config fields (and optional 'preset')")
    r.add_argument("--override", default=None, help="inline JSON overrides")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out-csv", required=True)
    r.add_argument("--videos", default=None)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--device", default="cuda")
    r.add_argument("--raw-dir", default=None, help="save post-ILP graphs here (or read them with --pp-only)")
    r.add_argument("--cache-dir", default=None,
                   help="also save the rich cache (weak peaks, gated logits, features) for offline experiments")
    r.add_argument("--pp-only", action="store_true", help="skip the models; repair cached raw graphs")
    r.add_argument("--time-budget-h", type=float, default=0.0,
                   help="per-shard wall-clock budget; beyond the projection the rest runs without TTA (0 = off)")
    m = sub.add_parser("merge")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--data-dir", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
