"""Sweep HOT dataset update parameters to find optimal values.

This script varies the update-related thresholds (``UPT``, ``UPH``,
``INTER`` and ``MB``) for the HOT dataset.  For each parameter
combination a copy of the base configuration is written to
``experiments/mcitrack/auto_update/<config>/config.yaml`` and evaluated
using :mod:`tracking.evaluate_hot`.

The results (precision@20 and AUC) are logged to
``experiments/mcitrack/auto_update/update_sweep_log.csv`` and a summary
table is printed when the sweep is complete.
"""

from __future__ import annotations

import csv
import itertools
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

try:  # allow both ``python -m`` and script execution
    from . import evaluate_hot  # type: ignore
except ImportError:  # pragma: no cover - fallback when run as script
    import evaluate_hot  # type: ignore


# Base configuration used for all experiments
BASE_CONFIG = Path("experiments/mcitrack/mcitrack_b224.yaml")
TRACKER = "mcitrack"

# Where to store auto-generated configs and logs
CONFIG_ROOT = Path("experiments/mcitrack/auto_update")
LOG_PATH = CONFIG_ROOT / "update_sweep_log.csv"

# Parameter search ranges
UPT_VALUES = [0.7, 0.8, 0.9]
UPH_VALUES = [0.9, 0.95]
INTER_VALUES = [10, 25, 50]
MB_VALUES = [200, 500]


def ensure_log() -> None:
    """Create the CSV log with header if it does not yet exist."""

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "config",
                    "UPT",
                    "UPH",
                    "INTER",
                    "MB",
                    "precision@20",
                    "AUC",
                ]
            )


def write_config(exp_dir: Path, overrides: dict) -> None:
    """Create a config.yaml in ``exp_dir`` with HOT overrides."""

    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg.setdefault("TEST", {})
    test_cfg.setdefault("UPT", {})["HOT"] = overrides["UPT"]
    test_cfg.setdefault("UPH", {})["HOT"] = overrides["UPH"]
    test_cfg.setdefault("INTER", {})["HOT"] = overrides["INTER"]
    test_cfg.setdefault("MB", {})["HOT"] = overrides["MB"]

    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def run_sweep(*, skip_run: bool = False, skip_vid: bool = True) -> None:
    """Execute the parameter sweep."""

    ensure_log()

    combos = list(itertools.product(UPT_VALUES, UPH_VALUES, INTER_VALUES, MB_VALUES))

    results: list[tuple] = []
    completed: set[str] = set()
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg_name = row["config"]
                completed.add(cfg_name)
                results.append(
                    (
                        cfg_name,
                        float(row["UPT"]),
                        float(row["UPH"]),
                        int(row["INTER"]),
                        int(row["MB"]),
                        float(row["precision@20"]),
                        float(row["AUC"]),
                    )
                )

    for upt, uph, inter, mb in tqdm(combos, desc="Experiments"):
        cfg_name = f"upt{upt}_uph{uph}_inter{inter}_mb{mb}"
        if cfg_name in completed:
            continue

        exp_dir = CONFIG_ROOT / cfg_name
        write_config(exp_dir, {"UPT": upt, "UPH": uph, "INTER": inter, "MB": mb})

        # Parameter name relative to experiments/mcitrack
        param_name = f"auto_update/{cfg_name}"
        dp20, auc = evaluate_hot.evaluate(
            TRACKER,
            param_name,
            skip_run=skip_run,
            skip_vid=skip_vid,
        )

        results.append((cfg_name, upt, uph, inter, mb, dp20, auc))

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    cfg_name,
                    upt,
                    uph,
                    inter,
                    mb,
                    dp20,
                    auc,
                ]
            )

    # Present a sorted summary
    results.sort(key=lambda r: (r[-2], r[-1]), reverse=True)
    print("\nSummary (sorted by precision@20 then AUC):")
    header = f"{'config':40} | UPT | UPH | INTER | MB | dp20 | AUC"
    print(header)
    print("-" * len(header))
    for cfg_name, upt, uph, inter, mb, dp20, auc in results:
        print(
            f"{cfg_name:40} | {upt:3.2f} | {uph:3.2f} | {inter:5d} | {mb:3d} | {dp20:5.3f} | {auc:5.3f}"
        )


if __name__ == "__main__":
    run_sweep()

