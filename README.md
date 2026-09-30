# Demand Forecasting, Segmentation & Inventory Replenishment

An end-to-end supply-chain analytics pipeline: weekly demand forecasting with
walk-forward backtesting, ABC-XYZ product segmentation, and segment-aware
inventory policy design (safety stock, reorder points, order-up-to levels)
validated by simulation against a uniform 98%-service policy.

> **Honest data note:** all demand data in this project is **synthetic**
> (generated with a seeded RNG — trend, annual seasonality, promotions, and
> per-SKU noise). It is built to mirror the structure of real retail/wholesale
> demand so the methodology can be demonstrated end to end; it is not actual
> sales history.

## Results (from the verified pipeline run)

**Forecasting** — category-level walk-forward backtest, 26 test weeks × 5 categories:

| Model | MAPE | RMSE (units) |
|---|---|---|
| Seasonal-naive (baseline) | 30.55% | 2,031 |
| SARIMA(1,1,1)(1,1,0,52) | 27.31% | 1,927 |
| **Holt-Winters ETS (best)** | **24.63%** | **1,491** |

Best model improves MAPE by **19.4% vs the seasonal-naive baseline**.
(Per category, SARIMA wins on Personal Care and Frozen Foods; Holt-Winters wins
overall. SARIMA underperforms the naive benchmark on Snacks — reported as-is in
`outputs/forecast_accuracy.csv`.)

**Segmentation** — 45 SKUs classified ABC (demand value) × XYZ (demand CV):

| | X | Y | Z |
|---|---|---|---|
| **A** (78.7% of value) | 9 | 5 | 2 |
| **B** (15.9% of value) | 6 | 6 | 3 |
| **C** (5.4% of value) | 10 | 4 | 0 |

**Inventory simulation** — 26-week order-up-to simulation vs actual demand:

| Policy | Avg inventory (units) | Fill rate |
|---|---|---|
| Uniform (98% service, 2-wk review for all) | 79,058 | 99.97% |
| **Segmented (ABC-XYZ policy matrix)** | **65,584** | **99.81%** |

The segmented policy cuts average inventory by **17.0%** while keeping the
demand-weighted fill rate at **99.8%**, above the 98% service target. It does so
by concentrating protection on high-value/forecastable SKUs (AX: 99% service,
weekly review) and deliberately de-stocking erratic low-value tails (BZ/CZ:
92%/85% service) instead of chasing unforecastable demand with safety stock.

## Methodology

1. **Data** (`src/generate_data.py`) — 45 SKUs (9 × 5 categories), 156 weeks of
   weekly demand with per-SKU trend, category seasonal peaks, multiplicative
   noise spanning CV ≈ 0.1–1.3, and ~6% promotion weeks (+15–75% lift).
2. **SQL rollups** (`sql/demand_rollups.sql`, run on SQLite via
   `src/sql_rollups.py`) — weekly/monthly category demand, trailing-52-week
   SKU value (ABC input), category value share, promo vs regular demand.
3. **Forecasting** (`src/forecast.py`) — expanding-window 1-step backtests over
   the last 26 weeks. Category level: seasonal-naive vs Holt-Winters vs
   SARIMA(1,1,1)(1,1,0,52) (fit once, rolled forward with fixed parameters;
   Prophet is used automatically if installed). SKU level: seasonal-naive
   residuals give a per-SKU forecast-error σ that feeds safety stock.
4. **Segmentation** (`src/segmentation.py`) — ABC by trailing-52-week demand
   value (A ≤ 80%, B ≤ 95%, C rest); XYZ by weekly-demand CV (X ≤ 0.5,
   Y ≤ 1.0, Z above).
5. **Inventory policy** (`src/inventory_policy.py`) — periodic-review
   order-up-to (R, S) policies: SS = z·σₑ·√(LT+R), ROP = mean·LT + z·σₑ·√LT.
   The uniform baseline uses z(98%) = 2.054, R = 2 for all SKUs; the segmented
   policy applies a 3×3 service/review matrix. Both are simulated against
   actual test-window demand, and the segmented plan is translated into
   category-level weekly replenishment and safety-stock totals (S&OP view,
   in `outputs/simulation_summary.json`).

## Tech stack

Python · pandas · NumPy · statsmodels (SARIMAX, Holt-Winters) · SciPy ·
matplotlib · SQLite (stdlib `sqlite3`) · Prophet (optional, auto-detected)

## How to run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_all.py     # ~80 seconds end to end
```

Each stage can also be run standalone, e.g. `.venv/bin/python src/forecast.py`.

## Repository structure

```
data/        generated demand + SKU master CSVs
sql/         demand_rollups.sql (SQLite dialect)
src/         generate_data · sql_rollups · forecast · segmentation ·
             inventory_policy · run_all
outputs/     forecast_accuracy.csv · forecast_predictions_category.csv
             sku_forecast_errors.csv · segmentation.csv
             inventory_policy.csv · simulation_detail.csv
             simulation_summary.json · sql_rollup_category_weekly.csv
             3 charts (forecast vs actual, ABC-XYZ heatmap, inventory sim)
```

## Limitations

- Synthetic data; the 17% inventory saving demonstrates the policy mechanics,
  not a real-world result.
- SARIMA is benchmarked at category level only (SKU-level SARIMA × 45 series
  is not needed to size safety stock here and keeps runtime low); SKU-level
  error σ comes from the seasonal-naive backtest.
- Simulation assumes lost sales, order-up-to reviews aligned to week 0, and
  pipelines starting empty with on-hand initialized at the order-up-to level.
