"""Score saved runs against gov_test with the same metrics as the baselines.

Items are matched to gov_test by row position, not item_id: award IDs such as
"0001" repeat across agencies, so item_id is not a unique key. Every run writes
its items in gov_test order (LLM runs score the first n rows when --limit is set).

Usage:  python scripts/score_run.py S1 [S2 ...] [--first-n 50]
        --first-n restricts every run to the first n rows, so a full-test baseline
        can be compared with a limited LLM run on identical items.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import RUNS, score     # noqa: E402
from src.data import load_gov                      # noqa: E402
from src.emissions import FactorLookup             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--first-n", type=int, default=None)
    args = ap.parse_args()

    test = load_gov("gov_test").reset_index(drop=True)
    lookup = FactorLookup(ROOT / "data" / "raw")
    for run in args.runs:
        data = json.loads((RUNS / f"{run}.json").read_text())
        items = data["items"]
        n = min(len(items), args.first_n or len(items))
        sub = test.head(n)
        ids = [it["item_id"] for it in items[:n]]
        if ids != sub["item_id"].tolist():
            sys.exit(f"{run}: items are not in gov_test row order; cannot score by position")
        pred = [str(it.get("naics6", it.get("pred", ""))) for it in items[:n]]
        suffix = f"_first{n}" if args.first_n else ""
        score(f"{run}{suffix}_scored", sub, pred, lookup,
              {"model": data.get("model"), "cost_usd": data.get("cost_usd")})


if __name__ == "__main__":
    main()
