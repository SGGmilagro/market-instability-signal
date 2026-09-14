WITH market_daily_tokens AS (
    SELECT
        DATE(block_time)                   AS trade_date,
        condition_id,
        question,
        asset_id,
        token_outcome_name,
        SUM(price * shares) / SUM(shares)  AS vwap_price,
        SUM(amount)                        AS daily_volume,
        COUNT(*)                           AS trade_count,
        STDDEV(price)                      AS price_std,
        MAX(price) - MIN(price)            AS price_range
    FROM polymarket_polygon.market_trades
    WHERE block_month >= DATE '2022-01-01'
      AND block_time  >= TIMESTAMP '2022-01-01 00:00:00'
      AND (
            LOWER(question) LIKE '%iran%'
         OR LOWER(question) LIKE '%hormuz%'
         OR LOWER(question) LIKE '%tehran%'
         OR LOWER(question) LIKE '%khamenei%'
         OR LOWER(question) LIKE '%bitcoin%'
         OR LOWER(question) LIKE '%btc%'
         OR LOWER(question) LIKE '%ethereum%'
         OR LOWER(question) LIKE '%crypto%'
         OR LOWER(question) LIKE '%solana%'
         OR LOWER(question) LIKE '%xrp%'
         OR LOWER(question) LIKE '%federal reserve%'
         OR LOWER(question) LIKE '%rate cut%'
         OR LOWER(question) LIKE '%recession%'
         OR LOWER(question) LIKE '%inflation%'
         OR LOWER(question) LIKE '%fomc%'
         OR LOWER(question) LIKE '%gdp%'
         OR LOWER(question) LIKE '%cpi%'
         OR LOWER(question) LIKE '%taiwan%'
         OR LOWER(question) LIKE '%china%'
         OR LOWER(question) LIKE '%russia%'
         OR LOWER(question) LIKE '%ukraine%'
         OR LOWER(question) LIKE '%north korea%'
         OR LOWER(question) LIKE '%nato%'
         OR LOWER(question) LIKE '%crude oil%'
         OR LOWER(question) LIKE '%oil price%'
         OR LOWER(question) LIKE '%wti%'
         OR LOWER(question) LIKE '%opec%'
         OR LOWER(question) LIKE '%natural gas%'
         OR LOWER(question) LIKE '%energy price%'
         OR LOWER(question) LIKE '%pandemic%'
         OR LOWER(question) LIKE '%covid%'
         OR LOWER(question) LIKE '%mpox%'
         OR LOWER(question) LIKE '%virus outbreak%'
      )
      AND LOWER(question) NOT LIKE '%nba%'
      AND LOWER(question) NOT LIKE '%nfl%'
      AND LOWER(question) NOT LIKE '%nhl%'
      AND LOWER(question) NOT LIKE '%mlb%'
      AND LOWER(question) NOT LIKE '%tennis%'
      AND LOWER(question) NOT LIKE '%golf%'
      AND LOWER(question) NOT LIKE '%fifa%'
      AND LOWER(question) NOT LIKE '%premier league%'
      AND LOWER(question) NOT LIKE '%champions league%'
      AND LOWER(question) NOT LIKE '%formula 1%'
      AND LOWER(question) NOT LIKE '%nascar%'
      AND LOWER(question) NOT LIKE '%boxing%'
      AND LOWER(question) NOT LIKE '%ufc%'
      AND LOWER(question) NOT LIKE '%esports%'
      AND LOWER(question) NOT LIKE '%oscar%'
      AND LOWER(question) NOT LIKE '%grammy%'
    GROUP BY DATE(block_time), condition_id, question, asset_id, token_outcome_name
    HAVING COUNT(*) >= 2
),

-- YES/NO consistency: sum of token VWAPs per market per day should be ~1.0
consistency AS (
    SELECT
        trade_date,
        condition_id,
        SUM(vwap_price)        AS vwap_sum,
        COUNT(DISTINCT asset_id) AS token_count
    FROM market_daily_tokens
    GROUP BY trade_date, condition_id
),

-- Pick highest-volume token (YES side) per market per day
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY trade_date, condition_id
            ORDER BY daily_volume DESC
        ) AS rn
    FROM market_daily_tokens
)

SELECT
    r.trade_date,
    r.condition_id,
    r.question,
    r.vwap_price,
    r.daily_volume,
    r.trade_count,
    r.price_std,
    r.price_range,
    c.vwap_sum                              AS yes_no_sum,
    ABS(c.vwap_sum - 1.0)                  AS consistency_error,
    CASE WHEN ABS(c.vwap_sum - 1.0) <= 0.05
         THEN TRUE ELSE FALSE END           AS is_consistent,
    CASE WHEN r.price_std <= 0.15
         THEN TRUE ELSE FALSE END           AS is_liquid
FROM ranked r
JOIN consistency c
  ON r.trade_date   = c.trade_date
 AND r.condition_id = c.condition_id
WHERE r.rn = 1
ORDER BY r.trade_date, r.condition_id
