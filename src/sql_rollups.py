"""Run sql/demand_rollups.sql against the generated CSVs using SQLite.

Loads data/sku_demand_weekly.csv and data/sku_master.csv into an in-memory
SQLite database, executes every query in sql/demand_rollups.sql, saves the
weekly category rollup to outputs/sql_rollup_category_weekly.csv, and prints a
short summary of each result.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "sql" / "demand_rollups.sql"


def _parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the .sql file into (name, statement) pairs using -- QUERY: markers."""
    queries: list[tuple[str, str]] = []
    name: str | None = None
    buf: list[str] = []
    for line in sql_text.splitlines():
        if line.strip().startswith("-- QUERY:"):
            if name is not None:
                queries.append((name, "\n".join(buf).strip().rstrip(";")))
            name = line.strip().split(":", 1)[1].strip()
            buf = []
        elif line.strip().startswith("--") or not line.strip():
            continue
        else:
            buf.append(line)
    if name is not None:
        queries.append((name, "\n".join(buf).strip().rstrip(";")))
    return queries


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    demand = pd.read_csv(data_dir / "sku_demand_weekly.csv")
    master = pd.read_csv(data_dir / "sku_master.csv")

    conn = sqlite3.connect(":memory:")
    demand.to_sql("sku_demand_weekly", conn, index=False)
    master.to_sql("sku_master", conn, index=False)

    queries = _parse_queries(SQL_PATH.read_text())
    for name, statement in queries:
        result = pd.read_sql_query(statement, conn)
        print(f"[sql_rollups] {name}: {result.shape[0]} rows x {result.shape[1]} cols")
        if name == "weekly_category_demand":
            path = out_dir / "sql_rollup_category_weekly.csv"
            result.to_csv(path, index=False)
            print(f"[sql_rollups]   -> saved {path}")
        elif name == "category_value_share":
            print(result.to_string(index=False))
        elif name == "promo_vs_regular":
            print(result.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
