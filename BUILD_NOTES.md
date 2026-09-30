# BUILD_NOTES

## What was built

A complete, runnable portfolio project at
`~/workspace/github-projects/demand-forecasting-inventory-replenishment/`
(**not** pushed to GitHub — left for the parent agent).

- `src/generate_data.py` — seeded (seed=42) synthetic data: 45 SKUs
  (Beverages, Snacks, Frozen Foods, Household, Personal Care × 9), 156 weeks
  from 2022-01-03, trend + two-harmonic annual seasonality + lognormal noise
  (target CVs 0.12–1.30) + 6% promo weeks (+15–75% lift). Writes
  `data/sku_demand_weekly.csv` (7,020 rows) and `data/sku_master.csv`
  (unit price, lead time 1–4 weeks).
- `sql/demand_rollups.sql` + `src/sql_rollups.py` — 5 SQLite queries (weekly
  category rollup, monthly rollup with MoM window function, trailing-52w SKU
  value, category value share, promo vs regular) executed via stdlib sqlite3;
  weekly rollup saved to `outputs/sql_rollup_category_weekly.csv`.
- `src/forecast.py` — walk-forward expanding-window 1-step backtest over the
  last 26 weeks. Category level: seasonal-naive, Holt-Winters ETS, SARIMA
  (1,1,1)(1,1,0,52); Prophet behind try/except import. SKU level:
  seasonal-naive for all 45 SKUs → residual σ per SKU. Writes
  `forecast_accuracy.csv`, `forecast_predictions_category.csv`,
  `sku_forecast_errors.csv`, `chart_forecast_vs_actual.png` (top-value SKU
  PCA-08).
- `src/segmentation.py` — ABC by trailing-52w demand value (80/95 cuts),
  XYZ by CV (0.5/1.0 cuts); `segmentation.csv` + `chart_abc_xyz_heatmap.png`.
- `src/inventory_policy.py` — (R,S) order-up-to policy design
  (SS = z·σₑ·√(LT+R), ROP reported), uniform 98%/R=2 baseline vs ABC-XYZ
  policy matrix; weekly simulation vs actual test demand; `inventory_policy.csv`,
  `simulation_detail.csv`, `simulation_summary.json` (incl. S&OP category
  translation), `chart_inventory_policy_comparison.png`.
- `src/run_all.py` — runs all five stages; `requirements.txt`; `README.md`.

## Run command (verified)

```bash
cd ~/workspace/github-projects/demand-forecasting-inventory-replenishment
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib statsmodels
.venv/bin/python src/run_all.py
```

End-to-end runtime: **~81 seconds** (under the ~3-minute budget).

## Verified headline results

- Forecast accuracy (category level, 130 backtest obs): naive MAPE **30.55%**,
  SARIMA **27.31%**, Holt-Winters **24.63%** → best-model improvement
  **+19.4% vs seasonal-naive**. SKU-level naive pooled MAPE 62.48% (1,170 obs).
- ABC-XYZ counts: AX 9, AY 5, AZ 2, BX 6, BY 6, BZ 3, CX 10, CY 4, CZ 0.
  ABC value shares: A 78.7%, B 15.9%, C 5.4% (boundary SKU rule; target 80/15/5).
- Inventory simulation (26 wks): uniform avg **79,058 units @ 99.97% fill**;
  segmented avg **65,584 units @ 99.81% fill** → **−17.04% average inventory**
  with fill still above the 98% target. Lowest segment fill: BZ 92.2%
  (by design — 92% segment target; aggregate is demand-weighted).

## Deviations, simplifications, limitations

- **Prophet skipped**: not installed in this environment; code path exists
  (try/except import, refit every 4 origins) and the skip is logged at runtime
  and noted in the README. Seasonal models carry the benchmark.
- **SARIMA scope**: fit at category level (5 series) per the brief's runtime
  allowance — fit once at the first origin with `concentrate_scale=True`
  (plain fit took ~29 s/series; this took ~4.6 s with an equivalent model),
  then rolled forward via `.apply(..., refit=False)`.
- **SARIMA weakness disclosed**: on Snacks, SARIMA (52.8% MAPE) loses to the
  naive baseline (42.9%); Holt-Winters is the best model overall. Kept honest
  in outputs/README rather than tuned away.
- **Policy design note**: a first pass (z by ABC, review by XYZ only) made
  the segmented policy *worse* (+2.0% inventory) because erratic Y/Z segments
  accumulate huge safety stock. Final design is a full 3×3 service/review
  matrix (99→85% service, 1→4-week review) that deliberately de-stocks
  erratic tails — this produced the −17.0% result. Documented in code + README.
- **Simulation assumptions**: lost sales (no backlog carry-over), on-hand
  initialized at order-up-to level, empty pipeline, review clock aligned at
  week 0 for every SKU; cycle-service z vs achieved fill are not identical
  measures (achieved fill exceeded targets because σₑ comes from 1-step naive
  errors, which are conservative).
- ABC boundary: A captured 78.7% rather than exactly 80% of value because the
  crossing SKU is assigned by cumulative-share cut.
- `.venv/` is inside the project dir (git-ignored via `.gitignore`); data and
  outputs are regenerated deterministically by `run_all.py`.
- Not pushed to GitHub (per tasking — parent agent handles the push).
