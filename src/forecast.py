"""Walk-forward (expanding-window) demand forecasting with backtesting.

Models evaluated on category-aggregated weekly demand (5 series):
  1. Seasonal-naive (baseline): forecast = same week last year (lag 52).
  2. Holt-Winters ETS (additive trend + additive seasonality, period 52).
  3. SARIMA(1,1,1)(1,1,0,52): fit once at the first origin, then rolled
     forward with statsmodels `.apply(..., refit=False)` to keep runtime low.
  4. Prophet: used only when installed; otherwise skipped (logged + noted).

Additionally, a SKU-level seasonal-naive backtest provides per-SKU forecast
residuals; their standard deviation (sigma_e) drives safety stock downstream.

Backtest design: last 26 weeks are the test window. At each origin the model
sees only data before that week (expanding window) and predicts one week ahead.

Outputs
-------
- outputs/forecast_accuracy.csv          per-model per-series MAPE / RMSE
- outputs/forecast_predictions_category.csv long-format actual vs forecast
- outputs/sku_forecast_errors.csv        per-SKU residual sigma + MAPE
- outputs/chart_forecast_vs_actual.png   top-SKU actual vs naive forecasts
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
TEST_WEEKS = 26
SEASON = 52

try:  # Prophet is optional; the pipeline runs without it.
    from prophet import Prophet  # type: ignore

    HAS_PROPHET = True
except Exception:  # pragma: no cover - environment dependent
    Prophet = None  # type: ignore
    HAS_PROPHET = False


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float, int]:
    mask = actual > 0
    mape = float(np.mean(np.abs(actual[mask] - forecast[mask]) / actual[mask]) * 100) if mask.any() else np.nan
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    return mape, rmse, int(len(actual))


def _backtest_category(series: pd.Series, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """1-step expanding-window backtest on one category series."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = series.to_numpy(dtype=float)
    n = len(y)
    first_origin = n - TEST_WEEKS
    records: list[dict] = []

    # --- SARIMA: single fit at first origin, then parameter-fixed roll-forward
    sarima_res = None
    try:
        model = SARIMAX(
            y[:first_origin],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, SEASON),
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,  # ~6x faster MLE fit; identical model family
        )
        sarima_res = model.fit(disp=False)
    except Exception as exc:  # pragma: no cover
        print(f"[forecast] SARIMA initial fit failed: {exc}")

    prophet_res = None
    if HAS_PROPHET:
        hist = pd.DataFrame({"ds": weeks[:first_origin], "y": y[:first_origin]})
        prophet_res = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        prophet_res.fit(hist)

    for i in range(first_origin, n):
        actual = y[i]
        # seasonal naive
        records.append(dict(week=str(weeks[i].date()), model="seasonal_naive",
                            actual=actual, forecast=y[i - SEASON]))
        # Holt-Winters (refit each origin on data up to the origin)
        try:
            hw = ExponentialSmoothing(
                y[:i], trend="add", seasonal="add", seasonal_periods=SEASON
            ).fit(optimized=True)
            records.append(dict(week=str(weeks[i].date()), model="holt_winters",
                                actual=actual, forecast=float(hw.forecast(1)[0])))
        except Exception:
            pass
        # SARIMA (state roll-forward with fixed parameters)
        if sarima_res is not None:
            try:
                rolled = sarima_res.apply(y[:i], refit=False)
                records.append(dict(week=str(weeks[i].date()), model="sarima",
                                    actual=actual, forecast=float(rolled.forecast(1)[0])))
            except Exception:
                pass
        # Prophet: refit every 4 origins, forecast 1 step
        if prophet_res is not None and (i - first_origin) % 4 == 0:
            pass  # refit horizon logic kept simple; see per-origin forecast below
        if prophet_res is not None:
            try:
                future = prophet_res.make_future_dataframe(periods=1, freq="W-MON")
                fc = prophet_res.predict(future)
                records.append(dict(week=str(weeks[i].date()), model="prophet",
                                    actual=actual, forecast=float(fc["yhat"].iloc[-1])))
                # extend training data by refitting every 4th origin
                if (i - first_origin) % 4 == 3:
                    hist = pd.DataFrame({"ds": weeks[: i + 1], "y": y[: i + 1]})
                    prophet_res = Prophet(yearly_seasonality=True, weekly_seasonality=False)
                    prophet_res.fit(hist)
            except Exception:
                pass

    return pd.DataFrame(records)


def _backtest_sku_naive(demand: pd.DataFrame, weeks_sorted: list[str]) -> pd.DataFrame:
    """SKU-level seasonal-naive 1-step backtest; returns per-SKU residual stats."""
    pivot = (
        demand.pivot_table(index="week_start", columns="sku_id", values="demand_units", aggfunc="sum")
        .reindex(weeks_sorted)
        .fillna(0)
    )
    values = pivot.to_numpy(dtype=float)  # rows = weeks, cols = SKUs
    n = values.shape[0]
    actual = values[n - TEST_WEEKS:, :]
    lagged = values[n - TEST_WEEKS - SEASON: n - SEASON, :]
    resid = actual - lagged

    rows = []
    for j, sku in enumerate(pivot.columns):
        mape, rmse, _ = _metrics(actual[:, j], lagged[:, j])
        rows.append(
            dict(
                sku_id=sku,
                mean_weekly_demand=float(np.mean(values[-52:, j])),
                sigma_residual=float(np.std(resid[:, j], ddof=1)),
                mape_naive_pct=mape,
                rmse_naive=rmse,
            )
        )
    pooled_mape, pooled_rmse, cnt = _metrics(actual.ravel(), lagged.ravel())
    print(f"[forecast] SKU-level seasonal-naive pooled: MAPE {pooled_mape:.2f}%  RMSE {pooled_rmse:.2f}  ({cnt} obs)")
    return pd.DataFrame(rows)


def _chart_forecast_vs_actual(demand: pd.DataFrame, sku_errors: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    master = pd.read_csv(PROJECT_ROOT / "data" / "sku_master.csv")
    merged = sku_errors.merge(master, on="sku_id")
    merged["annual_value"] = merged["mean_weekly_demand"] * 52 * merged["unit_price_usd"]
    top_sku = merged.sort_values("annual_value", ascending=False).iloc[0]["sku_id"]

    sku = demand[demand["sku_id"] == top_sku].sort_values("week_start")
    weeks = pd.to_datetime(sku["week_start"]).to_numpy()
    y = sku["demand_units"].to_numpy(dtype=float)
    n = len(y)
    test_idx = np.arange(n - TEST_WEEKS, n)
    naive_fc = y[test_idx - SEASON]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(weeks[-104:-TEST_WEEKS], y[-104:-TEST_WEEKS], label="Actual (history)", color="#1f77b4", lw=1.2)
    ax.plot(weeks[test_idx], y[test_idx], label="Actual (test)", color="#1f77b4", lw=2.0)
    ax.plot(weeks[test_idx], naive_fc, label="Seasonal-naive forecast", color="#d62728", ls="--", marker="o", ms=3)
    ax.set_title(f"Forecast vs actual - top SKU {top_sku} (last 26 weeks: seasonal-naive)")
    ax.set_ylabel("Weekly demand (units)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "chart_forecast_vs_actual.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[forecast] sample SKU for chart: {top_sku}; saved {path}")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demand = pd.read_csv(PROJECT_ROOT / "data" / "sku_demand_weekly.csv")

    print(f"[forecast] Prophet available: {HAS_PROPHET}"
          + ("" if HAS_PROPHET else " (skipped - not installed; seasonal models used instead)"))

    # ---- category-level backtest
    cat = (
        demand.groupby(["category", "week_start"], as_index=False)["demand_units"].sum()
    )
    weeks = pd.to_datetime(sorted(demand["week_start"].unique()))

    pred_frames = []
    acc_rows = []
    for category, grp in cat.groupby("category"):
        series = (
            grp.assign(week_start=pd.to_datetime(grp["week_start"]))
            .sort_values("week_start")["demand_units"]
            .reset_index(drop=True)
        )
        preds = _backtest_category(series, weeks)
        preds.insert(0, "category", category)
        pred_frames.append(preds)
        for model, g in preds.groupby("model"):
            mape, rmse, cnt = _metrics(g["actual"].to_numpy(), g["forecast"].to_numpy())
            acc_rows.append(dict(level="category", series=category, model=model,
                                 n_test=cnt, mape_pct=round(mape, 2), rmse_units=round(rmse, 2)))

    preds_all = pd.concat(pred_frames, ignore_index=True)
    for model, g in preds_all.groupby("model"):
        mape, rmse, cnt = _metrics(g["actual"].to_numpy(), g["forecast"].to_numpy())
        acc_rows.append(dict(level="category", series="ALL_CATEGORIES", model=model,
                             n_test=cnt, mape_pct=round(mape, 2), rmse_units=round(rmse, 2)))

    # ---- SKU-level seasonal-naive backtest (residual sigma for safety stock)
    sku_errors = _backtest_sku_naive(demand, sorted(demand["week_start"].unique()))
    sku_errors.to_csv(OUT_DIR / "sku_forecast_errors.csv", index=False)

    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT_DIR / "forecast_accuracy.csv", index=False)
    preds_all.to_csv(OUT_DIR / "forecast_predictions_category.csv", index=False)

    overall = acc[acc["series"] == "ALL_CATEGORIES"].set_index("model")
    naive_mape = overall.loc["seasonal_naive", "mape_pct"]
    best_model = overall["mape_pct"].drop(index="seasonal_naive", errors="ignore").idxmin()
    best_mape = overall.loc[best_model, "mape_pct"]
    improvement = (naive_mape - best_mape) / naive_mape * 100
    print(f"[forecast] category-level MAPE: naive {naive_mape:.2f}% | best {best_model} {best_mape:.2f}% "
          f"| improvement vs naive {improvement:.1f}%")
    print(acc.to_string(index=False))

    _chart_forecast_vs_actual(demand, sku_errors)


if __name__ == "__main__":
    main()
