"""ABC-XYZ product segmentation.

ABC  - by trailing-52-week demand value (units x unit price):
       A = SKUs accounting for the top 80% of value, B = next 15%, C = rest.
XYZ  - by coefficient of variation (CV) of weekly demand over the same window:
       X: CV <= 0.50 (stable), Y: 0.50 < CV <= 1.00, Z: CV > 1.00 (erratic).

Outputs
-------
- outputs/segmentation.csv            one row per SKU with segment labels
- outputs/chart_abc_xyz_heatmap.png   3x3 count heatmap annotated with value share
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

ABC_A_CUT = 0.80
ABC_B_CUT = 0.95
XYZ_X_CUT = 0.50
XYZ_Y_CUT = 1.00


def _abc_labels(value_sorted_desc: pd.Series) -> list[str]:
    cum_share = value_sorted_desc.cumsum() / value_sorted_desc.sum()
    return [
        "A" if c <= ABC_A_CUT else ("B" if c <= ABC_B_CUT else "C")
        for c in cum_share
    ]


def _xyz_label(cv: float) -> str:
    return "X" if cv <= XYZ_X_CUT else ("Y" if cv <= XYZ_Y_CUT else "Z")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demand = pd.read_csv(PROJECT_ROOT / "data" / "sku_demand_weekly.csv")
    master = pd.read_csv(PROJECT_ROOT / "data" / "sku_master.csv")

    weeks_sorted = sorted(demand["week_start"].unique())
    last52 = set(weeks_sorted[-52:])
    recent = demand[demand["week_start"].isin(last52)]

    stats = (
        recent.groupby("sku_id")["demand_units"]
        .agg(annual_units="sum", mean_w="mean", std_w="std")
        .reset_index()
    )
    stats["cv_52w"] = stats["std_w"] / stats["mean_w"].replace(0, np.nan)
    stats = stats.merge(master, on="sku_id", how="left")
    stats["annual_value_usd"] = stats["annual_units"] * stats["unit_price_usd"]

    stats = stats.sort_values("annual_value_usd", ascending=False).reset_index(drop=True)
    total_value = stats["annual_value_usd"].sum()
    stats["value_share_pct"] = (stats["annual_value_usd"] / total_value * 100).round(3)
    stats["cum_value_share_pct"] = (stats["annual_value_usd"].cumsum() / total_value * 100).round(2)
    stats["abc_class"] = _abc_labels(stats["annual_value_usd"])
    stats["xyz_class"] = stats["cv_52w"].apply(_xyz_label)
    stats["segment"] = stats["abc_class"] + stats["xyz_class"]

    out = stats[
        [
            "sku_id", "category", "annual_units", "annual_value_usd",
            "value_share_pct", "cum_value_share_pct", "abc_class",
            "cv_52w", "xyz_class", "segment",
        ]
    ].copy()
    out["cv_52w"] = out["cv_52w"].round(3)
    out_path = OUT_DIR / "segmentation.csv"
    out.to_csv(out_path, index=False)
    print(f"[segmentation] wrote {out_path} ({len(out)} SKUs)")

    counts = (
        out.pivot_table(index="abc_class", columns="xyz_class", values="sku_id",
                        aggfunc="count", fill_value=0)
        .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    )
    print("[segmentation] ABC-XYZ SKU counts:")
    print(counts.to_string())

    # ---- heatmap chart
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value_share = (
        out.pivot_table(index="abc_class", columns="xyz_class", values="annual_value_usd",
                        aggfunc="sum", fill_value=0)
        .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        / total_value * 100
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(counts.to_numpy(), cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i,
                f"{counts.iloc[i, j]} SKUs\n{value_share.iloc[i, j]:.1f}% value",
                ha="center", va="center", fontsize=10,
                color="white" if counts.iloc[i, j] > counts.to_numpy().max() / 2 else "black",
            )
    ax.set_xticks(range(3), ["X (stable)", "Y (moderate)", "Z (erratic)"])
    ax.set_yticks(range(3), ["A (top 80% value)", "B (next 15%)", "C (tail 5%)"])
    ax.set_title("ABC-XYZ segmentation: SKU counts and value share")
    fig.colorbar(im, ax=ax, label="SKU count")
    fig.tight_layout()
    chart_path = OUT_DIR / "chart_abc_xyz_heatmap.png"
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
    print(f"[segmentation] saved {chart_path}")


if __name__ == "__main__":
    main()
