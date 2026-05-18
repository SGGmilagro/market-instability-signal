import pandas as pd
import json
import subprocess

def build_dashboard():
    df = pd.read_csv("markets.csv")
    top15 = df.head(15).copy()
    top15 = top15.sort_values("instability_score", ascending=True)
    top15["market_short"] = top15["market"].str[:52] + "..."

    cluster_summary = df.groupby("cluster").agg(
        avg_score=("instability_score", "mean"),
        count=("market", "count"),
        contagion=("cluster_contagion", "max")
    ).reset_index().sort_values("avg_score", ascending=False)

    signals = df[df["alpha_signal"].isin(["LONG", "SHORT", "WATCH"])].head(12).copy()

    markets_json = top15.to_dict(orient="records")
    cluster_json = cluster_summary.to_dict(orient="records")
    signals_json = signals.to_dict(orient="records")
    all_json     = df.head(50).to_dict(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Market Instability Signal</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  :root {{
    --bg: #030712;
    --surface: #0d1117;
    --border: #1f2937;
    --red: #ef4444;
    --orange: #f97316;
    --yellow: #eab308;
    --green: #22c55e;
    --text: #f9fafb;
    --muted: #6b7280;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    padding: 2rem;
  }}

  header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .title-block h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
  }}

  .title-block h1 span {{ color: var(--red); }}
  .title-block p {{ color: var(--muted); font-size: 0.72rem; margin-top: 0.5rem; letter-spacing: 0.05em; text-transform: uppercase; }}

  .live-badge {{
    display: flex; align-items: center; gap: 0.5rem;
    background: #1a0a0a; border: 1px solid #7f1d1d;
    padding: 0.4rem 1rem; border-radius: 999px;
    font-size: 0.7rem; color: var(--red);
    letter-spacing: 0.1em; text-transform: uppercase;
  }}

  .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--red); animation: pulse 1.5s infinite; }}

  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
  }}

  .stats-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}

  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }}

  .stat-label {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }}
  .stat-value {{ font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; }}
  .stat-sub {{ font-size: 0.65rem; color: var(--muted); margin-top: 0.2rem; }}

  .grid-main {{
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}

  .card-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 1.25rem;
  }}

  canvas {{ max-height: 300px; }}

  .signal-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
  }}

  .signal-card {{
    border-radius: 8px;
    padding: 0.9rem 1rem;
    border-left: 3px solid;
  }}

  .signal-card.LONG   {{ border-color: var(--green);  background: #051a0e; }}
  .signal-card.SHORT  {{ border-color: var(--red);    background: #1a0505; }}
  .signal-card.WATCH  {{ border-color: var(--yellow); background: #1a1505; }}

  .signal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}

  .signal-badge {{
    font-size: 0.65rem; font-weight: 700;
    padding: 0.15rem 0.5rem; border-radius: 4px;
    letter-spacing: 0.1em;
  }}

  .LONG  .signal-badge {{ background: #14532d; color: #86efac; }}
  .SHORT .signal-badge {{ background: #7f1d1d; color: #fca5a5; }}
  .WATCH .signal-badge {{ background: #78350f; color: #fcd34d; }}

  .signal-cluster {{ font-size: 0.65rem; color: var(--muted); }}
  .signal-market  {{ font-size: 0.72rem; color: var(--text); margin-bottom: 0.5rem; line-height: 1.4; }}

  .signal-meta {{
    display: flex; gap: 1rem;
    font-size: 0.65rem; color: var(--muted);
  }}

  .dir-rising  {{ color: var(--red); }}
  .dir-falling {{ color: var(--green); }}
  .dir-flat    {{ color: var(--yellow); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.7rem; }}
  th {{
    text-align: left; color: var(--muted); font-weight: 400;
    letter-spacing: 0.08em; text-transform: uppercase;
    font-size: 0.62rem; padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 0.55rem 0.75rem; border-bottom: 1px solid #111827; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #111827; }}

  .score-pill {{
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 999px; font-size: 0.68rem; font-weight: 700;
  }}
  .score-high {{ background: #7f1d1d; color: #fca5a5; }}
  .score-med  {{ background: #78350f; color: #fcd34d; }}
  .score-low  {{ background: #14532d; color: #86efac; }}

  .alpha-badge {{
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 4px; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.05em;
  }}
  .alpha-LONG    {{ background: #14532d; color: #86efac; }}
  .alpha-SHORT   {{ background: #7f1d1d; color: #fca5a5; }}
  .alpha-WATCH   {{ background: #78350f; color: #fcd34d; }}
  .alpha-NEUTRAL {{ background: #1f2937; color: #6b7280; }}

  footer {{
    margin-top: 1.5rem; padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.62rem; color: var(--muted);
    text-align: center; letter-spacing: 0.05em;
  }}
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>Market <span>Instability</span> Signal</h1>
    <p>Entropy-weighted risk index · Polymarket on Polygon · On-chain prediction markets</p>
  </div>
  <div class="live-badge"><div class="dot"></div> Live Data</div>
</header>

<div class="stats-row" id="statsRow"></div>

<div class="grid-main">
  <div class="card" style="margin-bottom:0">
    <div class="card-title">Instability Score — Top 15 Markets</div>
    <canvas id="barChart"></canvas>
  </div>
  <div class="card" style="margin-bottom:0">
    <div class="card-title">Cluster Risk Heatmap</div>
    <canvas id="clusterChart"></canvas>
  </div>
</div>

<div style="margin-top:1.5rem" class="card">
  <div class="card-title">Alpha Signal Feed — LONG / SHORT / WATCH</div>
  <div class="signal-grid" id="signalGrid"></div>
</div>

<div class="card">
  <div class="card-title">Full Market Detail — Instability Index</div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Market</th>
        <th>Cluster</th>
        <th>Price</th>
        <th>Entropy</th>
        <th>Direction</th>
        <th>Liquidity (USDC)</th>
        <th>Fragility</th>
        <th>Score</th>
        <th>Signal</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

<footer>
  Data: Polymarket Gamma API + CLOB API (public, no auth) · Settlement: Polygon PoS ·
  Scoring: Shannon Entropy (35%) + Entropy Velocity (25%) + Liquidity Fragility (25%) + Volume Acceleration (15%)
</footer>

<script>
const markets  = {json.dumps(markets_json)};
const clusters = {json.dumps(cluster_json)};
const signals  = {json.dumps(signals_json)};
const allData  = {json.dumps(all_json)};

function scoreColor(s) {{
  if (s >= 60) return '#ef4444';
  if (s >= 45) return '#f97316';
  return '#eab308';
}}

// ── STATS ROW ────────────────────────────────────────────────────────────────
const statsRow   = document.getElementById('statsRow');
const topScore   = Math.max(...allData.map(m => m.instability_score));
const topMarket  = allData.find(m => m.instability_score === topScore);
const longCount  = allData.filter(m => m.alpha_signal === 'LONG').length;
const watchCount = allData.filter(m => m.alpha_signal === 'WATCH').length;
const topCluster = clusters[0]?.cluster || 'N/A';

const stats = [
  {{ label: 'Top Instability Score',  value: topScore.toFixed(1),  sub: topMarket?.market?.slice(0,35) + '...' }},
  {{ label: 'LONG Signals',           value: longCount,             sub: 'high entropy + rising + underpriced' }},
  {{ label: 'WATCH Signals',          value: watchCount,            sub: 'systemic or extreme entropy' }},
  {{ label: 'Hottest Cluster',        value: topCluster,            sub: `avg score ${{clusters[0]?.avg_score?.toFixed(1)}}` }},
];

stats.forEach(s => {{
  statsRow.innerHTML += `
    <div class="stat-card">
      <div class="stat-label">${{s.label}}</div>
      <div class="stat-value">${{s.value}}</div>
      <div class="stat-sub">${{s.sub}}</div>
    </div>`;
}});

// ── BAR CHART ────────────────────────────────────────────────────────────────
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {{
  type: 'bar',
  data: {{
    labels: markets.map(d => d.market_short),
    datasets: [{{
      data: markets.map(d => d.instability_score),
      backgroundColor: markets.map(d => scoreColor(d.instability_score)),
      borderRadius: 4,
      borderSkipped: false,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{ label: ctx => ` Score: ${{ctx.raw}}` }}
    }}}},
    scales: {{
      x: {{ grid: {{ color: '#1f2937' }}, ticks: {{ color: '#6b7280', font: {{ family: 'Space Mono', size: 10 }} }}, max: 100 }},
      y: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af', font: {{ family: 'Space Mono', size: 9 }} }} }}
    }}
  }}
}});

// ── CLUSTER CHART ────────────────────────────────────────────────────────────
const clusterCtx = document.getElementById('clusterChart').getContext('2d');
new Chart(clusterCtx, {{
  type: 'bar',
  data: {{
    labels: clusters.map(c => c.cluster),
    datasets: [{{
      label: 'Avg Instability Score',
      data: clusters.map(c => parseFloat(c.avg_score).toFixed(1)),
      backgroundColor: clusters.map(c => scoreColor(c.avg_score)),
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{ label: ctx => ` Avg Score: ${{ctx.raw}}` }}
    }}}},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af', font: {{ family: 'Space Mono', size: 10 }} }} }},
      y: {{ grid: {{ color: '#1f2937' }}, ticks: {{ color: '#6b7280', font: {{ family: 'Space Mono', size: 10 }} }}, max: 100 }}
    }}
  }}
}});

// ── SIGNAL FEED ──────────────────────────────────────────────────────────────
const signalGrid = document.getElementById('signalGrid');
if (signals.length === 0) {{
  signalGrid.innerHTML = '<p style="color:#6b7280;font-size:0.75rem">No LONG/SHORT/WATCH signals in current data.</p>';
}} else {{
  signals.forEach(s => {{
    const dirClass = `dir-${{s.entropy_direction}}`;
    const dirArrow = s.entropy_direction === 'rising' ? '↑' : s.entropy_direction === 'falling' ? '↓' : '→';
    signalGrid.innerHTML += `
      <div class="signal-card ${{s.alpha_signal}}">
        <div class="signal-header">
          <span class="signal-badge">${{s.alpha_signal}}</span>
          <span class="signal-cluster">${{s.cluster}} · contagion: ${{s.cluster_contagion}}</span>
        </div>
        <div class="signal-market">${{s.market}}</div>
        <div class="signal-meta">
          <span>price: ${{s.price.toFixed(2)}}</span>
          <span>entropy: ${{s.entropy}}</span>
          <span class="${{dirClass}}">${{dirArrow}} ${{s.entropy_direction}}</span>
          <span>score: ${{s.instability_score}}</span>
        </div>
      </div>`;
  }});
}}

// ── TABLE ────────────────────────────────────────────────────────────────────
const tbody = document.getElementById('tableBody');
allData.forEach((d, i) => {{
  const scoreClass = d.instability_score >= 60 ? 'score-high' : d.instability_score >= 45 ? 'score-med' : 'score-low';
  const dirArrow   = d.entropy_direction === 'rising'  ? '<span class="dir-rising">↑ rising</span>'
                   : d.entropy_direction === 'falling' ? '<span class="dir-falling">↓ falling</span>'
                   : '<span class="dir-flat">→ flat</span>';
  const liquidity  = d.liquidity != null ? `$${{parseFloat(d.liquidity).toLocaleString(undefined, {{maximumFractionDigits:0}})}}` : '—';
  const fragility  = d.liquidity_score != null ? parseFloat(d.liquidity_score).toFixed(3) : '—';

  tbody.innerHTML += `
    <tr>
      <td style="color:#6b7280">${{i+1}}</td>
      <td>${{d.market}}</td>
      <td style="color:#6b7280">${{d.cluster}}</td>
      <td>${{d.price.toFixed(2)}}</td>
      <td>${{d.entropy}}</td>
      <td>${{dirArrow}}</td>
      <td style="color:#6b7280">${{liquidity}}</td>
      <td style="color:#6b7280">${{fragility}}</td>
      <td><span class="score-pill ${{scoreClass}}">${{d.instability_score}}</span></td>
      <td><span class="alpha-badge alpha-${{d.alpha_signal}}">${{d.alpha_signal}}</span></td>
    </tr>`;
}});
</script>
</body>
</html>"""

    with open("dashboard.html", "w") as f:
        f.write(html)
    print("Dashboard saved — opening...")
    subprocess.run(["open", "dashboard.html"])

if __name__ == "__main__":
    build_dashboard()