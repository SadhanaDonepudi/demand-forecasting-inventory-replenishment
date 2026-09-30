"""End-to-end pipeline runner.

Order: generate data -> SQL rollups -> forecasting backtest -> segmentation
-> inventory policy + simulation. Produces every file under outputs/ and
prints the headline results.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# make sibling modules importable regardless of the caller's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    start = time.time()

    import generate_data
    import sql_rollups
    import forecast
    import segmentation
    import inventory_policy

    print("== 1/5 Generate synthetic demand data ==")
    generate_data.main()
    print("\n== 2/5 SQL demand rollups (SQLite) ==")
    sql_rollups.main()
    print("\n== 3/5 Walk-forward forecasting backtest ==")
    forecast.main()
    print("\n== 4/5 ABC-XYZ segmentation ==")
    segmentation.main()
    print("\n== 5/5 Inventory policy design + simulation ==")
    inventory_policy.main()

    print(f"\n[run_all] done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
