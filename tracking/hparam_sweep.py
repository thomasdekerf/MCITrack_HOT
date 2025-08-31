"""Run a hyper-parameter sweep for the HOT dataset evaluation.

Each experiment gets its own directory under ``experiments/mcitrack/auto``
containing a ``config.yaml`` file with the modified test settings.  The
evaluation always uses the base model checkpoint (no additional training is
required) and results are stored under ``results/<tracker>/auto/<exp_name>``.

The script logs the metrics for every combination to ``sweep_log.csv`` and
prints a summary table when finished.
"""

from __future__ import annotations

import itertools
import csv
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

try:  # allow both package and script execution
    from . import evaluate_hot  # type: ignore
except ImportError:  # pragma: no cover - fallback when run as script
    import evaluate_hot  # type: ignore


BASE_CONFIG = Path("experiments/mcitrack/mcitrack_b224.yaml")
TRACKER = "mcitrack"
CONFIG_ROOT = Path("experiments/mcitrack/auto")
LOG_PATH = CONFIG_ROOT / "sweep_log.csv"

# Parameter ranges for the sweep
SEARCH_SIZES = [224, 256]
SEARCH_FACTORS = [4.0, 3.0, 2.0]
TEMPLATE_SIZES = [112, 128]
TEMPLATE_FACTORS = [2.0, 1.5]
WINDOW_OPTIONS = [True, False]
NUM_TEMPLATES = [1, 5]


def ensure_log():
    """Create the log file with header if it does not yet exist."""
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "config",
                    "search_size",
                    "search_factor",
                    "template_size",
                    "template_factor",
                    "window",
                    "num_templates",
                    "precision@20",
                    "AUC",
                ]
            )


def write_config(exp_dir: Path, overrides: dict):
    """Create a config.yaml for this experiment with the given overrides."""
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply the overrides to TEST section
    cfg["TEST"].update(overrides)

    # Keep DATA settings in sync with TEST settings.  The model encoder is
    # constructed using the sizes specified under ``DATA``.  When we vary the
    # search/template sizes during a sweep, failing to update these fields means
    # the model is built with the default dimensions (e.g. template size 112),
    # while the tracker operates on the overridden sizes (e.g. 128).  This
    # mismatch leads to tensor shape errors during inference.  We therefore
    # mirror the relevant TEST overrides under DATA so that the network and
    # tracker agree on the spatial dimensions.

    data_cfg = cfg.setdefault("DATA", {})
    search_cfg = data_cfg.setdefault("SEARCH", {})
    template_cfg = data_cfg.setdefault("TEMPLATE", {})

    if "SEARCH_SIZE" in overrides:
        search_cfg["SIZE"] = overrides["SEARCH_SIZE"]
    if "SEARCH_FACTOR" in overrides:
        search_cfg["FACTOR"] = overrides["SEARCH_FACTOR"]
    if "TEMPLATE_SIZE" in overrides:
        template_cfg["SIZE"] = overrides["TEMPLATE_SIZE"]
    if "TEMPLATE_FACTOR" in overrides:
        template_cfg["FACTOR"] = overrides["TEMPLATE_FACTOR"]
    if "NUM_TEMPLATES" in overrides:
        template_cfg["NUMBER"] = overrides["NUM_TEMPLATES"]

    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def run_sweep(skip_run: bool = False, skip_vid: bool = True):
    ensure_log()

    combos = list(
        itertools.product(
            SEARCH_SIZES,
            SEARCH_FACTORS,
            TEMPLATE_SIZES,
            TEMPLATE_FACTORS,
            WINDOW_OPTIONS,
            NUM_TEMPLATES,
        )
    )

    results: list[tuple] = []
    completed = set()
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg_name = row["config"]
                completed.add(cfg_name)
                results.append(
                    (
                        cfg_name,
                        int(row["search_size"]),
                        float(row["search_factor"]),
                        int(row["template_size"]),
                        float(row["template_factor"]),
                        row["window"].lower() in ("true", "1"),
                        int(row.get("num_templates", 1)),
                        float(row["precision@20"]),
                        float(row["AUC"]),
                    )
                )

    for ss, sf, ts, tf, win, nt in tqdm(combos, desc="Experiments"):
        cfg_name = f"ss{ss}_sf{sf}_ts{ts}_tf{tf}_w{int(win)}_nt{nt}"
        if cfg_name in completed:
            continue

        exp_dir = CONFIG_ROOT / cfg_name

        write_config(
            exp_dir,
            {
                "SEARCH_SIZE": ss,
                "SEARCH_FACTOR": sf,
                "TEMPLATE_SIZE": ts,
                "TEMPLATE_FACTOR": tf,
                "WINDOW": bool(win),
                "NUM_TEMPLATES": nt,
            },
        )

        # Evaluate using the copied configuration.  The parameter name is the
        # experiment directory relative to experiments/mcitrack.
        param_name = f"auto/{cfg_name}"
        dp20, auc = evaluate_hot.evaluate(
            TRACKER,
            param_name,
            skip_run=skip_run,
            skip_vid=skip_vid,
        )

        results.append((cfg_name, ss, sf, ts, tf, win, nt, dp20, auc))

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    cfg_name,
                    ss,
                    sf,
                    ts,
                    tf,
                    win,
                    nt,
                    dp20,
                    auc,
                ]
            )

    # Present a simple table sorted by dp20
    results.sort(key=lambda r: (r[-2], r[-1]), reverse=True)
    print("\nSummary (sorted by precision@20 then AUC):")
    header = (
        f"{'config':35} | ss | sf  | ts | tf  | w | nt | dp20  | AUC"
    )
    print(header)
    print("-" * len(header))
    for cfg_name, ss, sf, ts, tf, win, nt, dp20, auc in results:
        print(
            f"{cfg_name:35} | {ss:3d} | {sf:3.1f} | {ts:3d} | {tf:3.1f} | {int(win):1d} | {nt:2d} | {dp20:5.3f} | {auc:5.3f}"
        )


if __name__ == "__main__":
    run_sweep()

