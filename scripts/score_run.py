"""Score a saved S-series run file against gov_test with the same metrics as the baselines.

Usage:  python scripts/score_run.py S1 [S2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import RUNS, score     # noqa: E402
from src.data import load_gov                      # noqa: E402
from src.emissions import FactorLookup             # noqa: E402


def main() -> None:
    test = load_gov("gov_test")
    lookup = FactorLookup(ROOT / "data" / "raw")
    for stage in sys.argv[1:]:
        data = json.loads((RUNS / f"{stage}.json").read_text())
        by_id = {it["item_id"]: it["naics6"] for it in data["items"]}
        sub = test[test["item_id"].isin(by_id)].reset_index(drop=True)
        pred = [by_id[i] for i in sub["item_id"]]
        score(f"{stage}_scored", sub, pred, lookup,
              {"model": data.get("model"), "cost_usd": data.get("cost_usd")})


if __name__ == "__main__":
    main()
