-- Demand rollups for the demand-forecasting / inventory-replenishment project.
-- Dialect: SQLite (executed by src/sql_rollups.py via Python's sqlite3 module).
-- Source tables (loaded from data/):
--   sku_demand_weekly(week_start TEXT, sku_id TEXT, category TEXT,
--                     demand_units INTEGER, promo_flag INTEGER)
--   sku_master(sku_id TEXT, category TEXT, unit_price_usd REAL,
--              lead_time_weeks INTEGER)

-- QUERY: weekly_category_demand
-- Weekly total demand by category (the series fed into category-level forecasting).
SELECT
    category,
    week_start,
    SUM(demand_units) AS total_units,
    COUNT(DISTINCT sku_id) AS active_skus,
    SUM(promo_flag) AS promo_sku_weeks
FROM sku_demand_weekly
GROUP BY category, week_start
ORDER BY category, week_start;

-- QUERY: monthly_category_rollup
-- Monthly demand rollup by category with month-over-month change.
SELECT
    category,
    substr(week_start, 1, 7) AS month,
    SUM(demand_units) AS monthly_units,
    ROUND(AVG(demand_units), 1) AS avg_weekly_units,
    SUM(demand_units) - LAG(SUM(demand_units)) OVER (
        PARTITION BY category ORDER BY substr(week_start, 1, 7)
    ) AS mom_change_units
FROM sku_demand_weekly
GROUP BY category, month
ORDER BY category, month;

-- QUERY: trailing_52w_sku_value
-- Trailing-52-week units and demand value by SKU (input to ABC classification).
WITH last52 AS (
    SELECT sku_id, category, demand_units
    FROM sku_demand_weekly
    WHERE week_start >= (SELECT date(MAX(week_start), '-357 days') FROM sku_demand_weekly)
)
SELECT
    l.sku_id,
    l.category,
    SUM(l.demand_units) AS annual_units,
    ROUND(SUM(l.demand_units) * m.unit_price_usd, 2) AS annual_value_usd,
    ROUND(AVG(l.demand_units), 2) AS mean_weekly_units,
    m.unit_price_usd,
    m.lead_time_weeks
FROM last52 l
JOIN sku_master m ON m.sku_id = l.sku_id
GROUP BY l.sku_id, l.category, m.unit_price_usd, m.lead_time_weeks
ORDER BY annual_value_usd DESC;

-- QUERY: category_value_share
-- Category share of total demand value over the full history.
SELECT
    d.category,
    SUM(d.demand_units) AS total_units,
    ROUND(SUM(d.demand_units * m.unit_price_usd), 2) AS total_value_usd,
    ROUND(
        100.0 * SUM(d.demand_units * m.unit_price_usd)
        / (SELECT SUM(d2.demand_units * m2.unit_price_usd)
           FROM sku_demand_weekly d2 JOIN sku_master m2 ON m2.sku_id = d2.sku_id),
        2
    ) AS value_share_pct
FROM sku_demand_weekly d
JOIN sku_master m ON m.sku_id = d.sku_id
GROUP BY d.category
ORDER BY total_value_usd DESC;

-- QUERY: promo_vs_regular
-- Promotion weeks vs regular weeks: average SKU-week demand by category.
SELECT
    category,
    CASE WHEN promo_flag = 1 THEN 'promo' ELSE 'regular' END AS week_type,
    COUNT(*) AS sku_weeks,
    ROUND(AVG(demand_units), 2) AS avg_units,
    SUM(demand_units) AS total_units
FROM sku_demand_weekly
GROUP BY category, week_type
ORDER BY category, week_type;
