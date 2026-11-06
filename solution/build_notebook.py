"""Build the Kaggle notebook biohub_clean.ipynb from biohub_clean.py."""
import json
import pprint
import sys
from pathlib import Path

HERE = Path(__file__).parent
module_src = (HERE / "biohub_clean.py").read_text()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "biohub_clean.ipynb"
# optional JSON with config overrides (the repair fixes chosen out of fold); injected into the CONFIG cell
FIXES = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}

INTRO = r"""# Biohub cell tracking: clean pipeline

A readable, bug-checked rebuild of the public 0.94x pipeline for the Biohub cell-tracking competition. It does the same work and is verified to reproduce the public reference graph edge for edge.

**What runs**
1. **Detection.** TemporalUNet3D from pilkwang's support pack, plus a second seed (seed314159) blended in with a retention guard.
2. **Association.** Node transformer with TTA-averaged features, harmonic forward/reverse fusion and a low-margin second-seed vote.
3. **Linking.** tracksdata ILP.
4. **Graph repair.** Motion relink, one-frame and two-frame gap closing, safe divisions confirmed by the DeepCenter prior, short-track filter and line-fit smoothing.

**What is removed compared with the public notebooks**
- The in-notebook validator and PP sweep. It tuned 7–20 knobs on 8 *training* videos that the weights were trained on, using the pre-patch division metric.
- Environment-variable configuration, string patching of the support-pack script, audit receipts and checksum guards.
- The recursive `rglob` over `/kaggle/input`, and dead switches: single-child repair, the division-geometry filter, and `GAP_CLOSE_MAX_GAP=2`, which the code silently capped at 1.
- Two repair micro-features with no measurable effect: the density-adaptive gap gate (±0.125 µm per step) and the adaptive short-track rescue. The official score on 8 training videos is identical with and without them.

Every model-level setting stays as in the public notebook: TTA views, second-seed weights, thresholds, ILP costs. On the 8 training videos the official score equals the public reference (0.9511). No in-notebook PP sweep means no run-time choice of `tight 5.5`.

**Known issue kept on purpose.** The public "8-view D4" TTA averages only 7 distinct views, because its anti-transpose equals the x-flip. The models were trained with flip augmentation only, never rotations. On training videos a true D4 set scores lower (0.9356) and flip-only sets score higher (0.9568–0.9600). That comparison is in-sample, so the public list stays until held-out validation decides. Set `tta` in the config to `d4`, `d4_7`, `flips8` or `flips4` to switch it.

**Inputs to attach:** the competition data, `pilkwang/biohub-tracking-support-pack-50ep-v1`, `pilkwang/biohub-temporal-unet3d-seed314159-v1` and `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`. Accelerator: GPU T4 x2, internet off.
"""

CONFIG = r'''import json, os
from pathlib import Path

def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError(f"none of these exist: {paths}")

COMP = first_existing("/kaggle/input/competitions/biohub-cell-tracking-during-development",
                      "/kaggle/input/biohub-cell-tracking-during-development")
def pack(slug):
    return first_existing(f"/kaggle/input/datasets/pilkwang/{slug}", f"/kaggle/input/{slug}")

SUPPORT = pack("biohub-tracking-support-pack-50ep-v1")
SECONDARY = pack("biohub-temporal-unet3d-seed314159-v1")
DEEPCENTER = pack("biohub-deepcenter-unet3d-center-prior-v1")
TEST_DIR = f"{COMP}/test"
WORK = Path("/kaggle/working")

CONFIG = {
    "preset": "clean",                 # "public_reference" reproduces the public 0.947 notebook exactly
    "support_pack": SUPPORT,
    "secondary_pack": SECONDARY,
    "deepcenter_ckpt": f"{DEEPCENTER}/weights/full_frame_center/best.pt",
    # Every other parameter keeps the Config default in biohub_clean.py (see the table in the module).
    # Repair fixes validated out of fold (leave-one-embryo-out), see README.md:
    **FIXES_JSON,
}
# Learned post-processing models (link re-scorer, division model) come from a private dataset; "{PP}" in a value
# is its mount point.
PP = next((p for p in ("/kaggle/input/datasets/kagglercs/biohub-pp-models", "/kaggle/input/biohub-pp-models")
           if Path(p).exists()), None)
for key, value in list(CONFIG.items()):
    if isinstance(value, str) and "{PP}" in value:
        if PP is None:
            raise FileNotFoundError("attach the kagglercs/biohub-pp-models dataset")
        CONFIG[key] = value.replace("{PP}", PP)
(WORK / "config.json").write_text(json.dumps(CONFIG, indent=1))
print(json.dumps(CONFIG, indent=1))
'''

DEPS = r'''# Offline dependencies from the support pack wheels (same package list as the public notebooks).
import importlib, importlib.util, re, subprocess, sys

SPECS = {
    "tracksdata": "tracksdata", "zarr": "zarr>=3.0.10,<4", "pyscipopt": "pyscipopt", "geff": "geff>=1.1.3.1.1",
    "geff_spec": "geff-spec<1.2", "ilpy": "ilpy>=0.5.1", "polars": "polars>=1.36", "blosc2": "blosc2",
    "dask": "dask", "imagecodecs": "imagecodecs", "skimage": "scikit-image>=0.24", "pyarrow": "pyarrow",
    "rustworkx": "rustworkx>=0.17.1", "sqlalchemy": "sqlalchemy>=2", "numcodecs": "numcodecs>=0.13,<0.16",
    "donfig": "donfig>=0.8", "google_crc32c": "google-crc32c>=1.5", "bidict": "bidict>=0.23.1",
    "psygnal": "psygnal>=0.14", "rich": "rich", "networkx": "networkx>=3.2.1", "pydantic": "pydantic>=2.11",
    "pydantic_core": "pydantic-core", "annotated_types": "annotated-types", "typing_extensions": "typing-extensions>=4.13",
    "typing_inspection": "typing-inspection", "markdown_it": "markdown-it-py", "pygments": "pygments",
    "click": "click", "cloudpickle": "cloudpickle", "fsspec": "fsspec", "partd": "partd", "locket": "locket",
    "toolz": "toolz", "yaml": "pyyaml", "ndindex": "ndindex", "msgpack": "msgpack", "numexpr": "numexpr",
    "deprecated": "deprecated", "wrapt": "wrapt", "imageio": "imageio", "PIL": "pillow", "tifffile": "tifffile",
    "lazy_loader": "lazy-loader", "tqdm": "tqdm",
}
EXTRA = {"polars": ["polars-runtime-32"]}
WHEELS = f"{SUPPORT}/wheels"

def polars_ok():
    try:
        import polars as pl
        from polars._plr import PySeries  # noqa: F401
        return hasattr(pl, "Float16")
    except Exception:
        return False

def zarr_ok():
    try:
        import zarr
        return int(str(zarr.__version__).split(".")[0]) >= 3
    except Exception:
        return False

def install(names):
    specs = []
    for n in names:
        specs += [SPECS[n], *EXTRA.get(n, [])]
    force = ["--force-reinstall"] if {"polars", "zarr"} & set(names) else []
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *force, "--find-links", WHEELS, *specs]
    print("installing", names)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    for mod in list(sys.modules):
        if mod.split(".")[0] in set(names) | {"tracksdata"}:
            sys.modules.pop(mod, None)

for _ in range(6):
    refresh = [n for n, ok in (("polars", polars_ok()), ("zarr", zarr_ok())) if not ok]
    missing = [n for n in SPECS if importlib.util.find_spec(n) is None]
    todo = list(dict.fromkeys(refresh + missing))
    if not todo:
        broken = []
        for n in SPECS:
            try:
                importlib.import_module(n)
            except Exception as exc:
                m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
                if m and m.group(1).split(".")[0] in SPECS:
                    broken.append(m.group(1).split(".")[0])
                else:
                    raise
        if not broken:
            break
        todo = broken
    install(todo)
print("dependencies ready")
'''

RUN = r'''import subprocess, sys, time, torch
t0 = time.time()
n_gpu = max(1, torch.cuda.device_count())
procs = []
for i in range(n_gpu):
    cmd = [sys.executable, str(WORK / "biohub_clean.py"), "run", "--config", str(WORK / "config.json"),
           "--data-dir", TEST_DIR, "--out-csv", str(WORK / f"part{i}.csv"), "--shard", str(i), "--nshards", str(n_gpu),
           "--time-budget-h", "10.5"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
    procs.append(subprocess.Popen(cmd, env=env, cwd=str(WORK)))
codes = [p.wait() for p in procs]
if any(codes):
    raise RuntimeError(f"shard exit codes {codes}")
subprocess.run([sys.executable, str(WORK / "biohub_clean.py"), "merge", "--parts",
                *[str(WORK / f"part{i}.csv") for i in range(n_gpu)], "--data-dir", TEST_DIR,
                "--out", str(WORK / "submission.csv")], check=True)
print(f"total {(time.time() - t0) / 60:.1f} min on {n_gpu} GPU(s)")
'''

SUMMARY = r'''import pandas as pd
stats = pd.concat([pd.read_json(WORK / f"part{i}.csv.stats.json") for i in range(n_gpu)], ignore_index=True)
print(stats[["dataset", "raw_nodes", "nodes", "edges", "divisions", "predict_s", "repair_s"]].to_string(index=False))
sub = pd.read_csv(WORK / "submission.csv")
print(sub.row_type.value_counts().to_dict(), "| datasets:", sub.dataset.nunique())
sub.head()
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


nb = {
    "cells": [
        md(INTRO),
        md("## 1. Configuration\nPaths are resolved without scanning `/kaggle/input`. Knob values live in `Config` inside the module."),
        code(CONFIG.replace("**FIXES_JSON,", "**" + pprint.pformat(FIXES, width=100, sort_dicts=True) + ",")),
        md("## 2. Dependencies\nInstalled offline from the support-pack wheels only when missing."),
        code(DEPS),
        md("## 3. Pipeline module\nThe whole pipeline in one file. It also runs locally for validation (`run`, `merge`, `--pp-only`)."),
        code("%%writefile /kaggle/working/biohub_clean.py\n" + module_src),
        md("## 4. Inference\nOne process per GPU; each processes every n-th test video and writes its own part. "
           "A time guard switches the remaining videos to single-view inference if a shard would exceed 10.5 h."),
        code(RUN),
        md("## 5. Summary"),
        code(SUMMARY),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")
