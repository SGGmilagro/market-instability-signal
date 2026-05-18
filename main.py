import requests
import pandas as pd
import math
import time
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────

NOISE_KEYWORDS = [
    "vs.", "spread:", "over/under", "nba", "nfl", "nhl", "mlb",
    "premier league", "fifa", "champions league", "super kings",
    "cricket", "tennis", "golf", "nascar", "f1", "formula",
    "will toulouse", "will arsenal", "will manchester", "will celtic",
    "will liverpool", "will real madrid", "will barcelona",
    "wrestle", "boxing", "ufc", "mma", "esports", "tweets from",
    "post 65", "post 90", "post 115", "academy award", "oscar",
    "grammy", "emmy", "box office",
    "bouzkova", "sabalenka", "swiatek", "djokovic", "alcaraz",
    "open:", "semifinal", "quarterfinal", "round of", "match:",
    "serie a", "bundesliga", "ligue 1", "eredivisie",
    "up or down on", "up or down this", "daily:", "spy (spy)", "spx)",
    "james bond", "announced as next", "will trump say", "henry cavill",
    "tom holland", "bret baier", "istanbul:", "pol martin"
]

CLUSTERS = {
    "Iran":         ["iran", "hormuz", "tehran", "persian", "khamenei", "irgc"],
    "Crypto":       ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp"],
    "Macro":        ["fed", "rate cut", "recession", "gdp", "inflation", "cpi", "fomc"],
    "Geopolitical": ["taiwan", "china", "russia", "ukraine", "north korea", "nato"],
    "Energy":       ["oil", "wti", "crude", "opec", "gas", "energy"],
    "Pandemic":     ["pandemic", "virus", "outbreak", "hantavirus", "covid", "mpox"],
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def is_relevant(question):
    q = question.lower()
    return not any(kw in q for kw in NOISE_KEYWORDS)

def get_cluster(question):
    q = question.lower()
    for cluster, keywords in CLUSTERS.items():
        if any(kw in q for kw in keywords):
            return cluster
    return "Other"

def compute_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def compute_liquidity_score(m):
    """
    Low liquidity = fragile market = instability amplifier.
    Returns 0 (deep/stable) to 1 (shallow/fragile).
    Inverted: less liquidity = higher fragility score.
    """
    try:
        liquidity = float(m.get("liquidity", 0))
        fragility = 1 / (1 + liquidity / 5000)
        return round(fragility, 4)
    except:
        return 0.5

# ── FETCH ────────────────────────────────────────────────────────────────────

def fetch_markets(limit=200):
    url = "https://gamma-api.polymarket.com/markets"
    all_markets = []
    offset = 0
    while len(all_markets) < limit:
        params = {
            "active": "true",
            "limit": 100,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false"
        }
        r = requests.get(url, params=params)
        batch = r.json()
        if not batch:
            break
        all_markets.extend(batch)
        offset += 100
    return all_markets[:limit]

def fetch_price_history(token_id):
    try:
        url = "https://clob.polymarket.com/prices-history"
        params = {
            "market": token_id,
            "interval": "1d",
            "fidelity": 60
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        history = data.get("history", [])
        if len(history) >= 2:
            return [float(h["p"]) for h in history]
    except:
        pass
    return []

# ── SCORE ────────────────────────────────────────────────────────────────────

def compute_entropy_velocity(price_history):
    if len(price_history) < 2:
        return 0, "flat"
    entropies = [compute_entropy(p) for p in price_history]
    mid    = entropies[len(entropies) // 2]
    recent = entropies[-1]
    velocity = recent - mid
    if velocity > 0.05:
        direction = "rising"
    elif velocity < -0.05:
        direction = "falling"
    else:
        direction = "flat"
    return round(velocity, 4), direction

def compute_volume_acceleration(m):
    try:
        vol_24h     = float(m.get("volume24hr", 0))
        vol_total   = float(m.get("volume", 1))
        days_active = max(float(m.get("daysActive", 7)), 1)
        avg_daily   = vol_total / days_active
        if avg_daily == 0:
            return 0
        acceleration = (vol_24h - avg_daily) / avg_daily
        return round(min(max(acceleration, -1), 1), 4)
    except:
        return 0

def generate_alpha_signal(entropy, entropy_velocity, direction, price, liquidity_score, vol_accel, cluster_contagion):
    # LONG: high uncertainty, flat or rising, crowd underpricing YES
    if (entropy > 0.75 and direction in ["rising", "flat"]
            and price < 0.55 and vol_accel >= 0):
        return "LONG"

    # SHORT: uncertainty resolving, market settling toward YES
    if (direction == "falling" and price > 0.65
            and entropy < 0.8):
        return "SHORT"

    # WATCH: systemic — multiple markets in same cluster unstable
    if cluster_contagion >= 4:
        return "WATCH"

    # WATCH: extremely high entropy even alone
    if entropy > 0.9:
        return "WATCH"

    return "NEUTRAL"

def score_markets(markets):
    rows = []
    for m in markets:
        if not is_relevant(m.get("question", "")):
            continue
        try:
            price           = float(m.get("lastTradePrice", 0.5))
            volume          = float(m.get("volume24hr", 0))
            cluster         = get_cluster(m.get("question", ""))
            entropy         = compute_entropy(price)
            vol_accel       = compute_volume_acceleration(m)
            liquidity_score = compute_liquidity_score(m)
            liquidity_raw   = float(m.get("liquidity", 0))

            token_id = None
            ctoken = m.get("clob_token_ids", None)
            if ctoken:
                try:
                    ids = json.loads(ctoken) if isinstance(ctoken, str) else ctoken
                    token_id = ids[0] if ids else None
                except:
                    pass

            rows.append({
                "market":           m.get("question", "N/A"),
                "price":            price,
                "volume_24hr":      volume,
                "entropy":          round(entropy, 4),
                "vol_accel":        vol_accel,
                "liquidity":        round(liquidity_raw, 2),
                "liquidity_score":  liquidity_score,
                "cluster":          cluster,
                "token_id":         token_id,
            })
        except:
            continue

    # cluster contagion count
    cluster_counts = {}
    for r in rows:
        if r["entropy"] > 0.75:
            cluster_counts[r["cluster"]] = cluster_counts.get(r["cluster"], 0) + 1

    # fetch price history for top 20 only
    print("Fetching price history for top 20 markets...")
    top_rows = sorted(rows, key=lambda x: x["entropy"], reverse=True)[:20]
    token_ids_top = {r["token_id"] for r in top_rows if r["token_id"]}

    history_cache = {}
    for tid in token_ids_top:
        if tid:
            history_cache[tid] = fetch_price_history(tid)
            time.sleep(0.1)

    final = []
    for r in rows:
        tid           = r["token_id"]
        price_history = history_cache.get(tid, []) if tid else []

        entropy_vel, direction = compute_entropy_velocity(price_history)
        cluster_contagion      = cluster_counts.get(r["cluster"], 0)

        entropy_norm    = r["entropy"]
        vel_norm        = min(max(entropy_vel + 0.5, 0), 1)
        vol_norm        = min(max(r["vol_accel"] + 1, 0), 2) / 2
        liquidity_norm  = r["liquidity_score"]

        instability = (
            entropy_norm   * 0.35 +
            vel_norm       * 0.25 +
            liquidity_norm * 0.25 +
            vol_norm       * 0.15
        )
        instability_score = round(instability * 100, 2)

        alpha = generate_alpha_signal(
            r["entropy"], entropy_vel, direction,
            r["price"], r["liquidity_score"],
            r["vol_accel"], cluster_contagion
        )

        final.append({
            "market":            r["market"],
            "price":             r["price"],
            "volume_24hr":       r["volume_24hr"],
            "entropy":           r["entropy"],
            "entropy_velocity":  entropy_vel,
            "entropy_direction": direction,
            "liquidity":         r["liquidity"],
            "liquidity_score":   r["liquidity_score"],
            "vol_acceleration":  r["vol_accel"],
            "cluster":           r["cluster"],
            "cluster_contagion": cluster_contagion,
            "instability_score": instability_score,
            "alpha_signal":      alpha,
        })

    df = pd.DataFrame(final)
    df = df.sort_values("instability_score", ascending=False).reset_index(drop=True)
    return df

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching markets...")
    markets = fetch_markets(200)
    print(f"Got {len(markets)} raw markets")

    df = score_markets(markets)

    print("\nTop 10 markets:")
    print(df[["market", "cluster", "liquidity", "entropy_direction", "alpha_signal", "instability_score"]].head(10))

    print("\nAlpha signals:")
    print(df["alpha_signal"].value_counts())

    print("\nCluster activity:")
    print(df.groupby("cluster")["instability_score"].mean().sort_values(ascending=False))

    df.to_csv("markets.csv", index=False)
    print("\nSaved to markets.csv")