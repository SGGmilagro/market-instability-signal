import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import math, json, requests, time
from datetime import datetime
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
CLUSTERS = {
    "Iran":         ["iran", "hormuz", "tehran", "persian", "khamenei", "irgc"],
    "Crypto":       ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp"],
    "Macro":        ["fed", "rate cut", "recession", "gdp", "inflation", "cpi", "fomc"],
    "Geopolitical": ["taiwan", "china", "russia", "ukraine", "north korea", "nato"],
    "Energy":       ["oil", "wti", "crude", "opec", "gas", "energy"],
    "Pandemic":     ["pandemic", "virus", "outbreak", "hantavirus", "covid", "mpox"],
}

NOISE_KEYWORDS = [
    "vs.", "spread:", "over/under", "nba", "nfl", "nhl", "mlb",
    "premier league", "fifa", "champions league", "cricket", "tennis",
    "golf", "nascar", "f1", "formula", "wrestle", "boxing", "ufc", "mma",
    "academy award", "oscar", "grammy", "emmy", "box office",
    "counter-strike", "lol:", "league of legends", "map handicap",
    "bo3", "bo5", "bo1", "nswc", "dota", "valorant", "cs2",
    "bouzkova", "sabalenka", "swiatek", "djokovic", "alcaraz",
    "open:", "semifinal", "quarterfinal", "round of",
    "serie a", "bundesliga", "ligue 1", "spy (spy)", "spx)",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def is_relevant(q):
    return not any(kw in q.lower() for kw in NOISE_KEYWORDS)

def get_cluster(q):
    q = q.lower()
    for cluster, kws in CLUSTERS.items():
        if any(kw in q for kw in kws):
            return cluster
    return None

def compute_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

# ── STEP 1: RECONSTRUCT ENTROPY TIME SERIES (volume-weighted) ─────────────────
def reconstruct_entropy_series():
    print("Fetching markets from API...")
    markets = []
    for offset in [0, 100, 200, 300, 400]:
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params={"active": "true", "limit": 100, "offset": offset,
                                 "order": "volume24hr", "ascending": "false"})
        batch = r.json()
        if not batch:
            break
        markets.extend(batch)

    rows = []
    for m in markets:
        q = m.get("question", "")
        if not is_relevant(q):
            continue
        cluster = get_cluster(q)
        if not cluster:
            continue
        ctoken = m.get("clobTokenIds", None)
        if not ctoken:
            continue
        try:
            ids = json.loads(ctoken) if isinstance(ctoken, str) else ctoken
            token_id = ids[0] if ids else None
        except:
            token_id = None
        if token_id:
            weight = float(m.get("volume24hr", 0) or 0)
            weight = max(weight, 1.0)
            rows.append({"token_id": token_id, "cluster": cluster,
                         "market": q, "weight": weight})

    df = pd.DataFrame(rows)
    print(f"Fetching price history for {len(df)} named-cluster markets...")

    cluster_daily = defaultdict(lambda: defaultdict(list))

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            r = requests.get(
                "https://clob.polymarket.com/prices-history",
                params={"market": row["token_id"], "interval": "max", "fidelity": 1440},
                timeout=10
            )
            history = r.json().get("history", [])
            for h in history:
                date = datetime.fromtimestamp(h["t"]).strftime("%Y-%m-%d")
                entropy = compute_entropy(float(h["p"]))
                cluster_daily[row["cluster"]][date].append((entropy, row["weight"]))
        except:
            pass
        time.sleep(0.05)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} done...")

    all_dates = sorted({d for c in cluster_daily.values() for d in c})
    records = []
    for date in all_dates:
        rec = {"date": date}
        for cluster in CLUSTERS:
            pairs = cluster_daily[cluster].get(date, [])
            if pairs:
                entropies = np.array([e for e, w in pairs])
                weights   = np.array([w for e, w in pairs])
                rec[f"{cluster}_entropy"] = np.average(entropies, weights=weights)
            else:
                rec[f"{cluster}_entropy"] = np.nan
        records.append(rec)

    entropy_df = pd.DataFrame(records)
    entropy_df["date"] = pd.to_datetime(entropy_df["date"])
    entropy_df = entropy_df.set_index("date").sort_index()
    cols = [f"{c}_entropy" for c in CLUSTERS]
    entropy_df["composite_entropy"] = entropy_df[cols].mean(axis=1)

    print("\nCluster market counts:")
    for cluster in CLUSTERS:
        n = sum(1 for _, row in df.iterrows() if row["cluster"] == cluster)
        print(f"  {cluster}: {n} markets")

    print(f"\nEntropy series built: {len(entropy_df)} days")
    return entropy_df

# ── STEP 2: FETCH BENCHMARKS ──────────────────────────────────────────────────
def fetch_benchmarks(start, end):
    print("\nFetching benchmarks from yfinance...")
    raw = {}
    for name, ticker in [("VIX", "^VIX"), ("BTC", "BTC-USD"), ("WTI", "CL=F")]:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            raw[name] = data["Close"].squeeze()
        except Exception as e:
            print(f"  Warning: {name} failed — {e}")

    bench = pd.DataFrame(raw)
    bench.index = pd.to_datetime(bench.index)

    for asset in ["BTC", "WTI"]:
        if asset in bench:
            log_ret = np.log(bench[asset] / bench[asset].shift(1))
            bench[f"{asset}_rvol30"] = log_ret.rolling(30).std() * np.sqrt(252)
            bench[f"{asset}_rvol7"]  = log_ret.rolling(7).std()  * np.sqrt(252)

    return bench

# ── STEP 3: ANALYSIS ──────────────────────────────────────────────────────────
def run_analysis(entropy_df, bench_df):
    merged = entropy_df.join(bench_df, how="inner")
    print(f"\nWindow: {merged.index[0].date()} → {merged.index[-1].date()} ({len(merged)} days)")

    pairs = [
        ("composite_entropy", "VIX",       "Composite entropy → VIX"),
        ("Macro_entropy",     "VIX",       "Macro cluster → VIX"),
        ("Crypto_entropy",    "BTC_rvol7", "Crypto cluster → BTC 7d realized vol"),
        ("Energy_entropy",    "WTI_rvol7", "Energy cluster → WTI 7d realized vol"),
    ]

    # ── Cross-correlation on levels ──
    print("\n── CROSS-CORRELATION on levels (entropy at T vs benchmark at T+lag) ──")
    for ecol, bcol, label in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            print(f"\n{label}: column missing, skipping")
            continue
        data = merged[[ecol, bcol]].dropna()
        if len(data) < 20:
            print(f"\n{label}: not enough data ({len(data)} rows)")
            continue
        x, y = data[ecol].values, data[bcol].values
        print(f"\n{label} (n={len(data)})")
        for lag in [0, 1, 2, 7]:
            xe, ye = (x[:-lag], y[lag:]) if lag > 0 else (x, y)
            r, p = pearsonr(xe, ye)
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
            print(f"  T+{lag}: r={r:+.3f}  p={p:.3f}  {sig}")

    # ── ADF stationarity test ──
    print("\n── ADF STATIONARITY TEST ──")
    for col in ["composite_entropy", "Macro_entropy", "Crypto_entropy",
                "VIX", "BTC_rvol7", "WTI_rvol7"]:
        if col not in merged.columns:
            continue
        series = merged[col].dropna()
        if len(series) < 10:
            print(f"  {col}: not enough data, skipping")
            continue
        result = adfuller(series, result_object=False)
        stat = "stationary" if result[1] < 0.05 else "NON-STATIONARY"
        print(f"  {col}: p={result[1]:.3f} → {stat}")

    # ── First-difference non-stationary series ──
    diff_cols = ["composite_entropy", "Macro_entropy", "Crypto_entropy",
                 "Energy_entropy", "BTC_rvol7", "WTI_rvol7"]
    for col in diff_cols:
        if col in merged.columns:
            merged[f"{col}_d"] = merged[col].diff()

    # VIX and BTC_rvol7 are stationary so test directly
    # Entropy series are non-stationary so use differenced versions
    pairs_granger = [
        ("composite_entropy_d", "VIX",       "Composite entropy Δ → VIX"),
        ("Macro_entropy_d",     "VIX",       "Macro cluster Δ → VIX"),
        ("Crypto_entropy_d",    "BTC_rvol7", "Crypto cluster Δ → BTC 7d realized vol"),
        ("Energy_entropy_d",    "WTI_rvol7", "Energy cluster Δ → WTI 7d realized vol"),
    ]

    # ── Granger causality ──
    print("\n── GRANGER CAUSALITY ──")
    print("   (entropy first-differenced where non-stationary; vol used directly where stationary)")
    for ecol, bcol, label in pairs_granger:
        if ecol not in merged.columns or bcol not in merged.columns:
            print(f"\n{label}: column missing, skipping")
            continue
        data = merged[[bcol, ecol]].dropna()
        if len(data) < 30:
            print(f"\n{label}: need 30+ days, have {len(data)}")
            continue
        try:
            print(f"\n{label} (n={len(data)})")
            res = grangercausalitytests(data, maxlag=3)
            for lag in [1, 2, 3]:
                F = res[lag][0]["ssr_ftest"][0]
                p = res[lag][0]["ssr_ftest"][1]
                sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
                print(f"  lag {lag}: F={F:.3f}  p={p:.3f}  {sig}")
        except Exception as e:
            print(f"  Granger failed: {e}")

    return merged

# ── STEP 4: PLOT ──────────────────────────────────────────────────────────────
def plot_results(merged):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Market Instability Signal — Entropy vs Volatility (90-day scaffold)", fontsize=12)

    plot_pairs = [
        ("composite_entropy", "Composite entropy index (volume-weighted)", "#E8593C"),
        ("VIX",               "VIX (implied vol)",                         "#3B8BD4"),
        ("BTC_rvol7",         "BTC 7d realized vol",                       "#9F7AEA"),
    ]

    for ax, (col, label, color) in zip(axes, plot_pairs):
        if col in merged.columns:
            ax.plot(merged.index, merged[col], color=color, linewidth=1.4, label=label)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(label, fontsize=9)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → backtest_results.png")
    plt.show()

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    entropy_df = reconstruct_entropy_series()
    entropy_df.to_csv("entropy_timeseries.csv")

    start = entropy_df.index.min().strftime("%Y-%m-%d")
    end   = entropy_df.index.max().strftime("%Y-%m-%d")
    bench_df = fetch_benchmarks(start, end)
    bench_df.to_csv("benchmarks.csv")

    merged = run_analysis(entropy_df, bench_df)
    merged.to_csv("merged_analysis.csv")

    plot_results(merged)

    print("\nFiles saved: entropy_timeseries.csv  benchmarks.csv  merged_analysis.csv  backtest_results.png")