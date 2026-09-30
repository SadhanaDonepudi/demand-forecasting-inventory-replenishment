"""Generate synthetic weekly SKU demand and SKU master data (seeded RNG).

Methodology
-----------
- 45 SKUs: 9 per category across 5 categories, 156 weeks (3 years) of weekly
  demand starting Monday 2022-01-03.
- Each SKU has a base volume, a small weekly growth trend, annual seasonality
  (fundamental + second harmonic, category-specific peak week), multiplicative
  lognormal noise sized to produce a spread of coefficients of variation
  (so ABC-XYZ segmentation has something interesting to find), and occasional
  one-week promotions (6% of SKU-weeks, +15% to +75% lift).
- Demand is rounded to whole units and floored at zero.

Outputs
-------
- data/sku_demand_weekly.csv : week_start, sku_id, category, demand_units, promo_flag
- data/sku_master.csv        : sku_id, category, unit_price_usd, lead_time_weeks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 156
START = "2022-01-03"  # a Monday
SKUS_PER_CATEGORY = 9
PROMO_PROB = 0.06

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# category: base median weekly units, seasonal amplitude, peak week of year,
# unit price range (USD), plausible lead times (weeks)
CATEGORIES = {
    "Beverages": dict(prefix="BEV", base=220, amp=0.35, peak=26, price=(1.20, 4.50), lead=(1, 2)),
    "Snacks": dict(prefix="SNK", base=300, amp=0.28, peak=48, price=(0.80, 3.50), lead=(1, 2)),
    "Frozen Foods": dict(prefix="FRZ", base=140, amp=0.30, peak=24, price=(2.50, 8.00), lead=(2, 3)),
    "Household": dict(prefix="HHD", base=120, amp=0.12, peak=10, price=(2.00, 9.00), lead=(2, 3, 4)),
    "Personal Care": dict(prefix="PCA", base=150, amp=0.16, peak=14, price=(1.50, 7.00), lead=(2, 3)),
}


def _draw_cv(rng: np.random.Generator) -> float:
    """Draw a target coefficient of variation, with a deliberately wide spread."""
    u = rng.random()
    if u < 0.60:
        return float(rng.uniform(0.12, 0.45))   # stable movers  -> XYZ = X
    if u < 0.85:
        return float(rng.uniform(0.45, 0.85))   # moderate       -> XYZ = Y
    return float(rng.uniform(0.85, 1.30))       # erratic        -> XYZ = Z


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    weeks = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS, dtype=float)
    year_len = 52.18  # average weeks per year

    demand_rows: list[dict] = []
    master_rows: list[dict] = []

    for cat, cfg in CATEGORIES.items():
        for j in range(SKUS_PER_CATEGORY):
            sku_id = f"{cfg['prefix']}-{j + 1:02d}"
            # lognormal volume factor -> a few A-class stars, a long C tail
            base = cfg["base"] * float(np.exp(rng.normal(0.0, 0.80)))
            growth = float(rng.uniform(-0.0008, 0.0035))       # weekly trend
            amp = cfg["amp"] * float(rng.uniform(0.70, 1.30))  # SKU-level jitter
            peak = cfg["peak"] + float(rng.uniform(-4, 4))
            cv = _draw_cv(rng)
            sigma = float(np.sqrt(np.log1p(cv**2)))            # lognormal sigma for target CV

            seasonal = (
                1.0
                + amp * np.cos(2 * np.pi * (t - peak) / year_len)
                + 0.30 * amp * np.cos(4 * np.pi * (t - peak) / year_len + 1.1)
            )
            trend = (1.0 + growth) ** t
            level = base * trend * seasonal

            noise = np.exp(sigma * rng.standard_normal(N_WEEKS) - 0.5 * sigma**2)
            promo_flag = (rng.random(N_WEEKS) < PROMO_PROB).astype(int)
            promo_lift = np.where(promo_flag == 1, rng.uniform(0.15, 0.75, N_WEEKS), 0.0)

            demand = level * noise * (1.0 + promo_lift)
            demand_units = np.maximum(0, np.rint(demand)).astype(int)

            for w, d, pf in zip(weeks, demand_units, promo_flag):
                demand_rows.append(
                    {
                        "week_start": w.date().isoformat(),
                        "sku_id": sku_id,
                        "category": cat,
                        "demand_units": int(d),
                        "promo_flag": int(pf),
                    }
                )
            master_rows.append(
                {
                    "sku_id": sku_id,
                    "category": cat,
                    "unit_price_usd": round(float(rng.uniform(*cfg["price"])), 2),
                    "lead_time_weeks": int(rng.choice(cfg["lead"])),
                }
            )

    demand_df = pd.DataFrame(demand_rows)
    master_df = pd.DataFrame(master_rows)
    return demand_df, master_df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demand_df, master_df = generate()
    demand_path = DATA_DIR / "sku_demand_weekly.csv"
    master_path = DATA_DIR / "sku_master.csv"
    demand_df.to_csv(demand_path, index=False)
    master_df.to_csv(master_path, index=False)
    print(
        f"[generate_data] wrote {demand_path} ({len(demand_df):,} rows, "
        f"{demand_df['sku_id'].nunique()} SKUs x {demand_df['week_start'].nunique()} weeks)"
    )
    print(f"[generate_data] wrote {master_path} ({len(master_df)} SKUs)")


if __name__ == "__main__":
    main()
