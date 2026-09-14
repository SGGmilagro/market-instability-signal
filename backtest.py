import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr, f as f_dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
CLUSTERS = {
    "Iran":         ["iran", "hormuz", "tehran", "persian", "khamenei", "irgc"],
    "Crypto":       ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp"],
    "Macro":        ["fed", "federal reserve", "rate cut", "recession", "gdp", "inflation", "cpi", "fomc"],
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
    "serie a", "bundesliga", "ligue 1",
]

THIN_CLUSTERS       = {"Energy", "Iran", "Pandemic"}
MIN_MARKETS_PER_DAY = 3

# Key events for event study
EVENTS = {
    "BTC Halving\n(Apr 2024)":      "2024-04-20",
    "Fed first cut\n(Sep 2024)":    "2024-09-18",
    "US Election\n(Nov 2024)":      "2024-11-05",
    "VIX spike\n(Aug 2025)":        "2025-08-05",
}

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

def safe_download(ticker, start, end):
    try:
        obj  = yf.Ticker(ticker)
        data = obj.history(start=start, end=end)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data["Close"].rename(ticker)
    except Exception as e:
        print(f"  Warning: {ticker} failed — {e}")
        return None

def hac_corr(x, y, nlags=None):
    """Pearson r with Newey-West HAC-corrected p-value."""
    n = len(x)
    if nlags is None:
        nlags = int(4 * (n / 100) ** (2 / 9))
    X = add_constant(x)
    try:
        res = OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": nlags, "use_correction": True})
        r   = float(np.corrcoef(x, y)[0, 1])
        return r, float(res.pvalues[1])
    except Exception:
        r, p = pearsonr(x, y)
        return r, p

# ── STEP 1: LOAD DUNE DATA ────────────────────────────────────────────────────
def load_dune_data(path="dune_data.csv", quality_filter=True):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print(f"  Raw rows: {len(df):,}")
    print(f"  Date range: {df['trade_date'].min().date()} → {df['trade_date'].max().date()}")

    df["cluster"] = df["question"].apply(
        lambda q: get_cluster(q) if is_relevant(q) else None
    )
    df = df[df["cluster"].notna()].copy()
    print(f"  After cluster filter: {len(df):,} rows")

    if quality_filter:
        main_mask = (
            (~df["cluster"].isin(THIN_CLUSTERS)) &
            (df["is_consistent"] == True) &
            (df["is_liquid"] == True)
        )
        thin_mask = (
            df["cluster"].isin(THIN_CLUSTERS) &
            (df["is_consistent"] == True)
        )
        df = df[main_mask | thin_mask].copy()
        print(f"  After adaptive quality filter: {len(df):,} rows")

    df["entropy"] = df["vwap_price"].apply(compute_entropy)

    print("\nCluster coverage:")
    for cluster in CLUSTERS:
        sub = df[df["cluster"] == cluster]
        print(f"  {cluster}: {sub['condition_id'].nunique()} markets, "
              f"{sub['trade_date'].nunique()} market-days")

    cluster_cols = []
    for cluster in CLUSTERS:
        sub = df[df["cluster"] == cluster]
        daily = sub.groupby("trade_date").apply(
            lambda g: np.average(g["entropy"], weights=g["daily_volume"])
            if g["daily_volume"].sum() > 0 and len(g) >= MIN_MARKETS_PER_DAY else np.nan
        ).rename(f"{cluster}_entropy")
        cluster_cols.append(daily)

    entropy_df = pd.concat(cluster_cols, axis=1, sort=True)
    entropy_df.index = pd.to_datetime(entropy_df.index)
    entropy_df = entropy_df.sort_index()
    entropy_df["composite_entropy"] = entropy_df[
        [f"{c}_entropy" for c in CLUSTERS]
    ].mean(axis=1)

    for col in ["composite_entropy", "Geopolitical_entropy", "Crypto_entropy",
                "Macro_entropy", "Energy_entropy", "Iran_entropy"]:
        if col in entropy_df.columns:
            entropy_df[f"inv_{col}"] = 1 - entropy_df[col]

    print(f"\nEntropy series: {len(entropy_df)} days")
    print(f"Non-null composite entropy days: {entropy_df['composite_entropy'].notna().sum()}")
    return entropy_df

# ── STEP 2: FETCH BENCHMARKS ──────────────────────────────────────────────────
def fetch_benchmarks(start, end):
    print("\nFetching benchmarks...")
    raw = {}
    for name, ticker in [("VIX", "^VIX"), ("BTC", "BTC-USD"), ("WTI", "USO")]:
        s = safe_download(ticker, start, end)
        if s is not None:
            raw[name] = s
            print(f"  {name} ({ticker}): {s.notna().sum()} days")

    derived = {}
    if "BTC" in raw:
        btc = raw["BTC"].dropna()
        lr  = np.log(btc / btc.shift(1))
        derived["BTC_rvol7"]  = lr.rolling(7).std()  * np.sqrt(252)
        derived["BTC_rvol30"] = lr.rolling(30).std() * np.sqrt(252)
        print(f"  BTC_rvol7 non-null: {derived['BTC_rvol7'].notna().sum()}")

    if "WTI" in raw:
        wti = raw["WTI"].dropna()
        ret = wti.pct_change()
        derived["WTI_rvol7"]  = ret.rolling(7).std()  * np.sqrt(252)
        derived["WTI_rvol30"] = ret.rolling(30).std() * np.sqrt(252)
        print(f"  WTI_rvol7 non-null: {derived['WTI_rvol7'].notna().sum()}")

    bench = pd.DataFrame({**raw, **derived})
    bench.index = pd.to_datetime(bench.index)
    return bench

# ── STEP 3: HAC CROSS-CORRELATION ─────────────────────────────────────────────
def run_hac_correlations(merged):
    print(f"\n{'='*60}")
    print("HAC/NEWEY-WEST CORRECTED CROSS-CORRELATIONS")
    print(f"{'='*60}")
    print("(Standard errors corrected for autocorrelation and heteroskedasticity)\n")

    pairs = [
        ("composite_entropy",     "VIX",       "Composite entropy → VIX"),
        ("Macro_entropy",         "VIX",       "Macro → VIX"),
        ("Crypto_entropy",        "BTC_rvol7", "Crypto → BTC 7d vol"),
        ("Energy_entropy",        "WTI_rvol7", "Energy → WTI 7d vol"),
        ("Geopolitical_entropy",  "VIX",       "Geopolitical → VIX"),
        ("Iran_entropy",          "WTI_rvol7", "Iran → WTI 7d vol"),
    ]

    for ecol, bcol, lbl in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            continue
        data = merged[[ecol, bcol]].dropna()
        if len(data) < 20:
            print(f"{lbl}: not enough data ({len(data)} rows)")
            continue
        x, y = data[ecol].values, data[bcol].values
        print(f"\n{lbl} (n={len(data)}, HAC corrected)")
        for lag in [0, 1, 2, 7, 10]:
            xe, ye = (x[:-lag], y[lag:]) if lag > 0 else (x, y)
            r, p = hac_corr(xe, ye)
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
            print(f"  T+{lag:2d}: r={r:+.3f}  p={p:.3f}  {sig}")

# ── STEP 4: VAR + IMPULSE RESPONSE ────────────────────────────────────────────
def run_var_irf(merged, horizon=10):
    print(f"\n{'='*60}")
    print("VAR + IMPULSE RESPONSE FUNCTIONS")
    print(f"{'='*60}")

    results = {}
    pairs = [
        ("Crypto_entropy", "BTC_rvol7", "Crypto entropy → BTC 7d vol"),
        ("Geopolitical_entropy", "VIX",  "Geopolitical entropy → VIX"),
    ]

    for ecol, bcol, lbl in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            continue
        data = merged[[ecol, bcol]].dropna()
        if len(data) < 50:
            print(f"\n{lbl}: insufficient data")
            continue
        try:
            model  = VAR(data)
            sel    = model.select_order(maxlags=10)
            lag_order = max(sel.aic, 1)
            result = model.fit(lag_order)
            irf    = result.irf(horizon)

            print(f"\n{lbl}")
            print(f"  Optimal lag (AIC): {lag_order}")
            print(f"  AIC: {result.aic:.4f}")

            # Durbin-Watson on residuals
            for i, col in enumerate([ecol, bcol]):
                dw = durbin_watson(result.resid.iloc[:, i])
                print(f"  Durbin-Watson ({col}): {dw:.3f}")

            results[lbl] = (irf, ecol, bcol, data)
        except Exception as e:
            print(f"\n{lbl}: VAR failed — {e}")

    return results

def plot_irf(var_results, filename="irf_results.png"):
    n = len(var_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (lbl, (irf, ecol, bcol, data)) in zip(axes, var_results.items()):
        horizon = irf.periods
        # IRF: response of bcol to shock in ecol
        ecol_idx = list(data.columns).index(ecol)
        bcol_idx = list(data.columns).index(bcol)

        irf_vals = irf.irfs[:, bcol_idx, ecol_idx]
        lower    = irf.stderr(orth=False)[:, bcol_idx, ecol_idx] if hasattr(irf, 'stderr') else None

        try:
            ci = irf.cum_effect_stderr(orth=False)
        except Exception:
            ci = None

        # Bootstrap confidence intervals
        try:
            err_bands = irf.errband_mc(orth=False, repl=200, signif=0.05)
            lower_band = err_bands[0][:, bcol_idx, ecol_idx]
            upper_band = err_bands[1][:, bcol_idx, ecol_idx]
            ax.fill_between(range(horizon + 1), lower_band, upper_band,
                            alpha=0.2, color="#E8593C")
        except Exception:
            pass

        ax.plot(range(horizon + 1), irf_vals, color="#E8593C", linewidth=2)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Days after shock")
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Impulse Response Functions — 1 std shock to entropy", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nIRF chart saved → {filename}")
    plt.show()

# ── STEP 5: OUT-OF-SAMPLE FORECAST ────────────────────────────────────────────
def run_out_of_sample(merged, split=0.70):
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE FORECAST TEST (train={int(split*100)}% / test={int((1-split)*100)}%)")
    print(f"{'='*60}")

    pairs = [
        ("Crypto_entropy", "BTC_rvol7", "BTC 7d realized vol"),
        ("Geopolitical_entropy", "VIX",  "VIX"),
    ]

    oos_results = {}
    for ecol, bcol, label in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            continue
        data = merged[[ecol, bcol]].dropna()
        n       = len(data)
        n_train = int(n * split)
        train   = data.iloc[:n_train]
        test    = data.iloc[n_train:]

        if len(test) < 20:
            print(f"\n{label}: test set too small ({len(test)} rows)")
            continue

        print(f"\n{label} (train n={n_train}, test n={len(test)})")

        # Baseline: AR(2) on benchmark alone — ARIMA(2,0,0) is standard citation
        from statsmodels.tsa.arima.model import ARIMA
        try:
            ar_fit      = ARIMA(train[bcol].values, order=(2, 0, 0)).fit()
            ar_forecast = ar_fit.forecast(steps=len(test))
            ar_rmse     = float(np.sqrt(np.mean(
                (test[bcol].values - np.array(ar_forecast)) ** 2)))
        except Exception as e:
            print(f"  AR baseline failed: {e}")
            ar_rmse = np.nan

        # VAR(2): benchmark + entropy
        try:
            var_fit      = VAR(train).fit(2)
            var_forecast = var_fit.forecast(train.values[-2:], steps=len(test))
            bcol_idx     = list(data.columns).index(bcol)
            var_pred     = var_forecast[:, bcol_idx]
            var_rmse     = float(np.sqrt(np.mean(
                (test[bcol].values - var_pred) ** 2)))
        except Exception as e:
            print(f"  VAR failed: {e}")
            var_rmse = np.nan

        improvement = ((ar_rmse - var_rmse) / ar_rmse * 100) if not np.isnan(ar_rmse) else np.nan
        print(f"  AR(2) RMSE:     {ar_rmse:.6f}")
        print(f"  VAR(2) RMSE:    {var_rmse:.6f}")
        print(f"  Improvement:    {improvement:+.2f}%")

        oos_results[label] = {
            "ar_rmse": ar_rmse, "var_rmse": var_rmse,
            "improvement_pct": improvement,
            "test": test, "var_pred": var_pred if not np.isnan(var_rmse) else None,
            "bcol": bcol
        }

    return oos_results

def plot_oos(oos_results, filename="oos_forecast.png"):
    n = len(oos_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, res) in zip(axes, oos_results.items()):
        test = res["test"]
        bcol = res["bcol"]
        ax.plot(test.index, test[bcol].values, color="white",
                linewidth=1.2, label="Actual", alpha=0.9)
        if res["var_pred"] is not None:
            ax.plot(test.index, res["var_pred"], color="#E8593C",
                    linewidth=1.2, linestyle="--", label="VAR forecast")
        ax.set_title(f"{label}\nVAR RMSE improvement: {res['improvement_pct']:+.2f}%",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Out-of-sample forecast: AR(2) baseline vs VAR with entropy", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"OOS forecast chart saved → {filename}")
    plt.show()

# ── STEP 6: CHOW STRUCTURAL BREAK TEST ────────────────────────────────────────
def run_chow_test(merged, breakdate="2025-01-01"):
    print(f"\n{'='*60}")
    print(f"CHOW STRUCTURAL BREAK TEST (breakdate: {breakdate})")
    print(f"{'='*60}")

    pairs = [
        ("Geopolitical_entropy", "VIX",       "Geopolitical → VIX"),
        ("Crypto_entropy",       "BTC_rvol7", "Crypto → BTC 7d vol"),
        ("composite_entropy",    "VIX",       "Composite → VIX"),
    ]

    for ecol, bcol, lbl in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            continue
        data = merged[[ecol, bcol]].dropna()
        bp   = pd.Timestamp(breakdate)
        pre  = data[data.index <  bp]
        post = data[data.index >= bp]

        if len(pre) < 10 or len(post) < 10:
            print(f"\n{lbl}: insufficient data in one segment")
            continue

        def ssr_ols(df):
            X = add_constant(df[ecol].values)
            m = OLS(df[bcol].values, X).fit()
            return m.ssr

        k   = 2
        n   = len(data)
        ssr_full = ssr_ols(data)
        ssr_pre  = ssr_ols(pre)
        ssr_post = ssr_ols(post)

        F = ((ssr_full - (ssr_pre + ssr_post)) / k) / \
            ((ssr_pre + ssr_post) / (n - 2 * k))
        p = 1 - f_dist.cdf(F, k, n - 2 * k)

        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
        print(f"\n{lbl}")
        print(f"  Pre:  n={len(pre)}  ({pre.index[0].date()} → {pre.index[-1].date()})")
        print(f"  Post: n={len(post)} ({post.index[0].date()} → {post.index[-1].date()})")
        print(f"  F={F:.3f}  p={p:.4f}  {sig}")
        if p < 0.05:
            print(f"  → Structural break confirmed at {breakdate}")
        else:
            print(f"  → No significant structural break at {breakdate}")

        # Regime-specific slopes
        for name, df_seg in [("Pre", pre), ("Post", post)]:
            X = add_constant(df_seg[ecol].values)
            m = OLS(df_seg[bcol].values, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5})
            print(f"  {name} slope: β={m.params[1]:.4f}  p={m.pvalues[1]:.3f}")

# ── STEP 7: EVENT STUDY ────────────────────────────────────────────────────────
def run_event_study(entropy_df, bench_df, window=10, filename="event_study.png"):
    print(f"\n{'='*60}")
    print("EVENT STUDY (entropy ±10 days around key events)")
    print(f"{'='*60}")

    merged = entropy_df.join(bench_df, how="inner")
    cols   = ["composite_entropy", "Geopolitical_entropy", "Crypto_entropy", "VIX", "BTC_rvol7"]
    cols   = [c for c in cols if c in merged.columns]

    n_events = len(EVENTS)
    n_cols   = len(["composite_entropy", "VIX", "BTC_rvol7"])
    fig, axes = plt.subplots(n_cols, n_events,
                             figsize=(5 * n_events, 4 * n_cols), sharey="row")

    plot_vars = [
        ("composite_entropy", "Composite entropy", "#E8593C"),
        ("VIX",               "VIX",               "#3B8BD4"),
        ("BTC_rvol7",         "BTC 7d vol",         "#9F7AEA"),
    ]

    for col_i, (event_label, event_date) in enumerate(EVENTS.items()):
        edate = pd.Timestamp(event_date)
        print(f"\n{event_label.replace(chr(10), ' ')} ({event_date})")
        for row_i, (var, var_label, color) in enumerate(plot_vars):
            ax = axes[row_i, col_i] if n_events > 1 else axes[row_i]
            if var not in merged.columns:
                continue
            window_data = merged[var].reindex(
                pd.date_range(edate - pd.Timedelta(days=window),
                              edate + pd.Timedelta(days=window), freq="D")
            )
            days = np.arange(-window, window + 1)
            vals = window_data.values
            ax.plot(days, vals, color=color, linewidth=1.5)
            ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_title(f"{event_label}" if row_i == 0 else "", fontsize=8)
            ax.set_ylabel(var_label if col_i == 0 else "", fontsize=8)
            ax.set_xlabel("Days from event" if row_i == n_cols - 1 else "")
            ax.grid(True, alpha=0.2)
            print(f"  {var_label}: pre-avg={np.nanmean(vals[:window]):.3f}  "
                  f"post-avg={np.nanmean(vals[window:]):.3f}")

    fig.suptitle("Event Study — Entropy and Volatility Around Key Events", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nEvent study chart saved → {filename}")
    plt.show()

# ── STEP 8: CLUSTER HEATMAP ────────────────────────────────────────────────────
def plot_cluster_heatmap(merged, filename="cluster_heatmap.png"):
    entropy_cols = [c for c in merged.columns
                    if "_entropy" in c and "inv_" not in c and "composite" not in c]
    bench_cols   = ["VIX", "BTC_rvol7", "WTI_rvol7"]
    bench_cols   = [c for c in bench_cols if c in merged.columns]

    all_cols  = entropy_cols + bench_cols
    corr_data = merged[all_cols].dropna()
    corr_mat  = corr_data.corr()

    labels = [c.replace("_entropy", "").replace("_rvol7", " vol7") for c in all_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < 0.5 else "white")

    ax.set_title("Cluster Entropy × Benchmark Correlation Matrix", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved → {filename}")
    plt.show()

# ── STEP 9: STANDARD ANALYSIS (Granger + ADF) ─────────────────────────────────
def run_analysis(entropy_df, bench_df, label=""):
    merged = entropy_df.join(bench_df, how="inner")
    print(f"\n{'='*60}")
    print(f"ANALYSIS {label}")
    print(f"Window: {merged.index[0].date()} → {merged.index[-1].date()} ({len(merged)} days)")
    print(f"{'='*60}")

    pairs = [
        ("composite_entropy",        "VIX",       "Composite → VIX"),
        ("Macro_entropy",            "VIX",       "Macro → VIX"),
        ("Crypto_entropy",           "BTC_rvol7", "Crypto → BTC 7d vol"),
        ("Energy_entropy",           "WTI_rvol7", "Energy → WTI 7d vol"),
        ("Geopolitical_entropy",     "VIX",       "Geopolitical → VIX"),
        ("Iran_entropy",             "WTI_rvol7", "Iran → WTI 7d vol"),
        ("inv_composite_entropy",    "VIX",       "Inv composite → VIX"),
        ("inv_Geopolitical_entropy", "VIX",       "Inv Geopolitical → VIX"),
        ("inv_Crypto_entropy",       "BTC_rvol7", "Inv Crypto → BTC 7d vol"),
    ]

    print("\n── CROSS-CORRELATION (HAC corrected) ──")
    for ecol, bcol, lbl in pairs:
        if ecol not in merged.columns or bcol not in merged.columns:
            continue
        data = merged[[ecol, bcol]].dropna()
        if len(data) < 20:
            print(f"\n{lbl}: not enough data ({len(data)} rows)")
            continue
        x, y = data[ecol].values, data[bcol].values
        print(f"\n{lbl} (n={len(data)})")
        for lag in [0, 1, 2, 7, 10]:
            xe, ye = (x[:-lag], y[lag:]) if lag > 0 else (x, y)
            r, p = hac_corr(xe, ye)
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
            print(f"  T+{lag:2d}: r={r:+.3f}  p={p:.3f}  {sig}")

    print("\n── ADF STATIONARITY TEST ──")
    non_stationary = []
    test_cols = [c for c in merged.columns
                 if "entropy" in c or c in ["VIX", "BTC_rvol7", "WTI_rvol7"]]
    for col in test_cols:
        series = merged[col].dropna()
        if len(series) < 10:
            print(f"  {col}: insufficient data")
            continue
        result = adfuller(series, result_object=False)
        stat = "stationary" if result[1] < 0.05 else "NON-STATIONARY"
        if result[1] >= 0.05:
            non_stationary.append(col)
        print(f"  {col}: p={result[1]:.3f} → {stat}")

    for col in non_stationary:
        if col in merged.columns:
            merged[f"{col}_d"] = merged[col].diff()

    print("\n── GRANGER CAUSALITY (maxlag=10) ──")
    for ecol, bcol, lbl in pairs:
        ecol_g = f"{ecol}_d" if ecol in non_stationary else ecol
        bcol_g = f"{bcol}_d" if bcol in non_stationary else bcol
        if ecol_g not in merged.columns or bcol_g not in merged.columns:
            continue
        data = merged[[bcol_g, ecol_g]].dropna()
        if len(data) < 30:
            print(f"\n{lbl}: insufficient data ({len(data)} rows)")
            continue
        try:
            print(f"\n{lbl} (n={len(data)})")
            res = grangercausalitytests(data, maxlag=10)
            for lag in [1, 2, 3, 5, 7, 10]:
                F = res[lag][0]["ssr_ftest"][0]
                p = res[lag][0]["ssr_ftest"][1]
                sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
                print(f"  lag {lag:2d}: F={F:.3f}  p={p:.3f}  {sig}")
        except Exception as e:
            print(f"  Granger failed: {e}")

    return merged

# ── STEP 10: MARKOV SWITCHING ─────────────────────────────────────────────────
def run_markov_regime_analysis(merged):
    print(f"\n{'='*60}")
    print("MARKOV SWITCHING REGIME ANALYSIS")
    print(f"{'='*60}")

    col = "Geopolitical_entropy"
    if col not in merged.columns:
        return merged

    series = merged[[col]].dropna()[col]
    if len(series) < 100:
        return merged

    print(f"\nFitting 2-regime Markov AR(1) on {col} (n={len(series)})...")
    try:
        model  = MarkovAutoregression(series, k_regimes=2, order=1, switching_ar=True)
        result = model.fit(disp=False, maxiter=200)
        probs  = result.smoothed_marginal_probabilities
        probs.index = series.index[len(series) - len(probs):]
        regime = probs.idxmax(axis=1)

        print("\nRegime summary:")
        for r in [0, 1]:
            r_dates  = regime[regime == r].index
            geo_mean = series.loc[r_dates].mean()
            vix_mean = merged.loc[r_dates, "VIX"].dropna().mean() \
                       if "VIX" in merged else np.nan
            print(f"  Regime {r}: {len(r_dates)} days | "
                  f"avg entropy={geo_mean:.3f} | avg VIX={vix_mean:.2f}")

        print("\nRegime-specific Granger (Geopolitical entropy → VIX):")
        for r in [0, 1]:
            r_dates = regime[regime == r].index
            r_data  = merged.loc[r_dates, [col, "VIX"]].dropna()
            if len(r_data) < 30:
                print(f"  Regime {r}: insufficient data ({len(r_data)} rows)")
                continue
            print(f"\n  Regime {r} (n={len(r_data)}):")
            try:
                res = grangercausalitytests(r_data[["VIX", col]], maxlag=3)
                for lag in [1, 2, 3]:
                    F = res[lag][0]["ssr_ftest"][0]
                    p = res[lag][0]["ssr_ftest"][1]
                    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "—")
                    print(f"    lag {lag}: F={F:.3f}  p={p:.3f}  {sig}")
            except Exception as e:
                print(f"    Granger failed: {e}")

        merged["geo_regime"] = np.nan
        merged.loc[regime.index, "geo_regime"] = regime.values
        probs.to_csv("regime_probabilities.csv")
        print("\nRegime probabilities → regime_probabilities.csv")

    except Exception as e:
        print(f"Markov model failed: {e}")

    return merged

# ── STEP 11: MAIN PLOT ────────────────────────────────────────────────────────
def plot_results(merged, filename="backtest_results_dune.png"):
    has_regime = "geo_regime" in merged.columns and merged["geo_regime"].notna().any()
    n_panels   = 6 if has_regime else 5
    fig, axes  = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
    fig.suptitle("Market Instability Signal — Entropy vs Volatility (Dune 2023–2026)",
                 fontsize=12)

    plot_pairs = [
        ("composite_entropy", "Composite entropy index (volume-weighted)", "#E8593C"),
        ("VIX",               "VIX (implied vol)",                         "#3B8BD4"),
        ("BTC_rvol7",         "BTC 7d realized vol",                       "#9F7AEA"),
        ("WTI_rvol7",         "WTI 7d realized vol (USO)",                 "#F6A623"),
    ]

    for ax, (col, lbl, color) in zip(axes[:4], plot_pairs):
        if col in merged.columns:
            ax.plot(merged.index, merged[col], color=color, linewidth=1.2, label=lbl)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_ylabel(lbl, fontsize=8)

    ax5 = axes[4]
    if "Geopolitical_entropy" in merged.columns and "VIX" in merged.columns:
        roll_r = (
            merged[["Geopolitical_entropy", "VIX"]].dropna()
            .rolling(90).corr().unstack()["Geopolitical_entropy"]["VIX"]
        )
        ax5.plot(roll_r.index, roll_r.values, color="#E8593C", linewidth=1.2,
                 label="90-day rolling r: Geopolitical entropy vs VIX")
        ax5.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax5.set_ylabel("Rolling r", fontsize=8)
        ax5.set_ylim(-0.6, 0.6)
        ax5.legend(loc="upper left", fontsize=9)
        ax5.grid(True, alpha=0.2)

    if has_regime and n_panels == 6:
        ax6 = axes[5]
        geo = merged["Geopolitical_entropy"].dropna()
        ax6.plot(geo.index, geo.values, color="#E8593C", linewidth=1.0,
                 alpha=0.8, label="Geopolitical entropy")
        for r, color in [(0, "#3B8BD4"), (1, "#F6A623")]:
            mask = merged["geo_regime"] == r
            ax6.fill_between(merged.index, 0, 1, where=mask, alpha=0.25,
                             color=color, transform=ax6.get_xaxis_transform(),
                             label=f"Regime {r}")
        ax6.set_ylabel("Geopolitical entropy", fontsize=8)
        ax6.legend(loc="upper left", fontsize=9)
        ax6.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nMain chart saved → {filename}")
    plt.show()

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    entropy_df = load_dune_data("dune_data.csv", quality_filter=True)
    entropy_df.to_csv("entropy_timeseries_dune.csv")

    start = entropy_df.index.min().strftime("%Y-%m-%d")
    end   = entropy_df.index.max().strftime("%Y-%m-%d")
    bench_df = fetch_benchmarks(start, end)
    bench_df.to_csv("benchmarks_dune.csv")

    # Standard analysis (Granger + ADF + HAC correlations)
    merged = run_analysis(entropy_df, bench_df, label="[MAIN — adaptive quality filter]")

    # Additional econometric tests
    run_hac_correlations(merged)
    run_chow_test(merged, breakdate="2025-01-01")

    # VAR + IRF
    var_results = run_var_irf(merged, horizon=10)
    plot_irf(var_results, "irf_results.png")

    # Out-of-sample forecast
    oos_results = run_out_of_sample(merged, split=0.70)
    plot_oos(oos_results, "oos_forecast.png")

    # Markov regime analysis
    merged = run_markov_regime_analysis(merged)

    # Visualizations
    merged.to_csv("merged_analysis_dune.csv")
    plot_results(merged, "backtest_results_dune.png")
    plot_cluster_heatmap(merged, "cluster_heatmap.png")
    run_event_study(entropy_df, bench_df, window=10, filename="event_study.png")

    # Robustness check
    print("\n\nRunning robustness check (no quality filter)...")
    entropy_raw = load_dune_data("dune_data.csv", quality_filter=False)
    bench_raw   = fetch_benchmarks(
        entropy_raw.index.min().strftime("%Y-%m-%d"),
        entropy_raw.index.max().strftime("%Y-%m-%d")
    )
    merged_raw = run_analysis(entropy_raw, bench_raw, label="[ROBUSTNESS — all markets]")

    print("\nDone. Files saved:")
    print("  entropy_timeseries_dune.csv   benchmarks_dune.csv")
    print("  merged_analysis_dune.csv      backtest_results_dune.png")
    print("  regime_probabilities.csv      irf_results.png")
    print("  oos_forecast.png              cluster_heatmap.png")
    print("  event_study.png")