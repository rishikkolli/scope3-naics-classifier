"""Print a markdown table of every run in eval/runs/ (and refresh docs/results.md).

Usage:  python scripts/summarize_runs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "eval" / "runs"


def main() -> None:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        if "metrics" in d:
            rows.append((d["run"], d.get("n"), d["metrics"]))
    keys: list[str] = []
    for _, _, m in rows:
        keys += [k for k in m if k not in keys]
    lines = ["| run | n | " + " | ".join(keys) + " |",
             "|---|---|" + "---|" * len(keys)]
    for run, n, m in rows:
        lines.append(f"| {run} | {n} | " + " | ".join(str(m.get(k, "")) for k in keys) + " |")
    table = "\n".join(lines)
    print(table)
    out = ROOT / "docs" / "results.md"
    out.write_text("# Run results\n\nRegenerate with `python scripts/summarize_runs.py`.\n\n"
                   + table + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
