import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0a0a0f; color: #e8e8e8; }

[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

.metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4a9eff; }
.metric-value { font-size: 2rem; font-weight: 800; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }
.metric-delta { font-size: 0.85rem; margin-top: 6px; font-family: 'Space Mono', monospace; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4a9eff;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.fear-badge  { background:#2d1515; color:#ff6b6b; border:1px solid #ff6b6b33; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'Space Mono',monospace; }
.greed-badge { background:#152d15; color:#6bff6b; border:1px solid #6bff6b33; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'Space Mono',monospace; }
.neutral-badge { background:#1e1e15; color:#ffd700; border:1px solid #ffd70033; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'Space Mono',monospace; }

.insight-box {
    background: #12121f;
    border-left: 3px solid #4a9eff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.strategy-box {
    background: #0f1a0f;
    border-left: 3px solid #6bff6b;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

div[data-testid="stMetric"] {
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-radius: 10px;
    padding: 14px;
}

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #c8c8d8 !important; }

.stSelectbox label, .stMultiSelect label, .stSlider label { color: #888 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }

hr { border-color: #1e1e2e !important; }
</style>
""", unsafe_allow_html=True)

# ── PLOT THEME ─────────────────────────────────────────────────────────────────
DARK_BG   = '#0a0a0f'
CARD_BG   = '#12121f'
FEAR_COL  = '#ff6b6b'
GREED_COL = '#6bff6b'
NEUT_COL  = '#ffd700'
BLUE      = '#4a9eff'
GRID_COL  = '#1e1e2e'

def dark_fig(figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors='#888', labelsize=9)
    ax.xaxis.label.set_color('#888')
    ax.yaxis.label.set_color('#888')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.7)
    return fig, ax

def dark_fig_multi(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=CARD_BG)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes_flat:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors='#888', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.7)
    return fig, axes

def sent_color(s):
    s = str(s).lower()
    if 'fear' in s:   return FEAR_COL
    if 'greed' in s:  return GREED_COL
    return NEUT_COL

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(trades_file, sentiment_file):
    trades_df    = pd.read_csv(trades_file)
    sentiment_df = pd.read_csv(sentiment_file)

    # ── Sentiment ──
    sentiment_df.columns = [c.strip().lower() for c in sentiment_df.columns]
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    if 'classification' in sentiment_df.columns:
        sentiment_df = sentiment_df.rename(columns={'classification': 'sentiment', 'sentiment': 'score'})
    sentiment_df['sentiment'] = sentiment_df['sentiment'].astype(str).str.strip().replace({
        'Extreme Fear': 'Fear', 'Extreme Greed': 'Greed'
    })

    # ── Trades ──
    trades_df.columns = [c.strip().lower().replace(' ', '_') for c in trades_df.columns]
    rename_map = {'closed_pnl':'pnl','size_usd':'size','execution_price':'price'}
    trades_df = trades_df.rename(columns=rename_map)
    trades_df['pnl']  = pd.to_numeric(trades_df['pnl'],  errors='coerce').fillna(0)
    trades_df['size'] = pd.to_numeric(trades_df['size'],  errors='coerce')
    trades_df['price']= pd.to_numeric(trades_df.get('price', pd.Series()), errors='coerce')
    trades_df['date'] = pd.to_datetime(trades_df['timestamp'], unit='ms').dt.normalize()
    trades_df['is_win']  = (trades_df['pnl'] > 0).astype(int)
    if 'side' in trades_df.columns:
        trades_df['is_long'] = trades_df['side'].str.upper().str.contains('B|BUY|LONG', na=False).astype(int)

    # ── Merge ──
    merged = trades_df.merge(sentiment_df[['date','sentiment']+(['score'] if 'score' in sentiment_df.columns else [])], on='date', how='left')
    merged = merged.dropna(subset=['sentiment'])

    # ── Daily trader aggregation ──
    account_col = 'account' if 'account' in merged.columns else merged.columns[0]
    agg = {'pnl':['sum','mean'], 'is_win':'mean', 'size':['mean','sum'], 'sentiment':'first'}
    if 'is_long' in merged.columns: agg['is_long'] = 'mean'
    daily = merged.groupby([account_col,'date']).agg(agg)
    daily.columns = ['_'.join(c) for c in daily.columns]
    daily = daily.reset_index().rename(columns={
        'pnl_sum':'daily_pnl','pnl_mean':'avg_pnl_per_trade',
        'is_win_mean':'win_rate','size_mean':'avg_size','size_sum':'total_volume',
        'sentiment_first':'sentiment','is_long_mean':'long_ratio'
    })
    tc = merged.groupby([account_col,'date']).size().reset_index(name='trade_count')
    daily = daily.merge(tc, on=[account_col,'date'])

    # Segments
    daily['perf_segment'] = pd.qcut(
        daily.groupby(account_col)['daily_pnl'].transform('sum'),
        q=[0,.33,.67,1.0], labels=['Loser','Neutral','Winner'], duplicates='drop'
    )
    daily['freq_segment'] = pd.qcut(
        daily['trade_count'], q=[0,.33,.67,1.0],
        labels=['Infrequent','Moderate','Frequent'], duplicates='drop'
    )

    return daily, account_col

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    trades_file    = st.file_uploader("Upload **trades.csv**",    type='csv', key='trades')
    sentiment_file = st.file_uploader("Upload **fear_greed.csv**", type='csv', key='sent')
    st.markdown("---")
    st.markdown("### Filters")

# ── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Trader Performance vs Market Sentiment")
st.markdown("*Hyperliquid × Fear & Greed Index — Primetrade.ai Assignment*")
st.markdown("---")

if not trades_file or not sentiment_file:
    st.info("👈 Upload both CSV files in the sidebar to begin.")
    st.markdown("""
    **Files needed:**
    - `trades.csv` — Hyperliquid historical trader data
    - `fear_greed.csv` — Bitcoin Fear & Greed Index
    """)
    st.stop()

# Load
with st.spinner("Processing data..."):
    daily, account_col = load_data(trades_file, sentiment_file)

# Sidebar filters
sentiments = st.sidebar.multiselect("Sentiment", options=daily['sentiment'].unique().tolist(),
                                     default=daily['sentiment'].unique().tolist())
date_min = daily['date'].min().date()
date_max = daily['date'].max().date()
date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max),
                                    min_value=date_min, max_value=date_max)

# Apply filters
filtered = daily[daily['sentiment'].isin(sentiments)]
if len(date_range) == 2:
    filtered = filtered[(filtered['date'] >= pd.Timestamp(date_range[0])) &
                        (filtered['date'] <= pd.Timestamp(date_range[1]))]

fear_d  = filtered[filtered['sentiment'] == 'Fear']
greed_d = filtered[filtered['sentiment'] == 'Greed']

# ── KPI CARDS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Overview Metrics</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

total_traders = filtered[account_col].nunique()
total_trades  = filtered['trade_count'].sum()
avg_wr        = filtered['win_rate'].mean()
fear_pnl      = fear_d['daily_pnl'].mean()  if len(fear_d)  else 0
greed_pnl     = greed_d['daily_pnl'].mean() if len(greed_d) else 0

k1.metric("Total Traders",   f"{total_traders:,}")
k2.metric("Total Trade Days", f"{total_trades:,.0f}")
k3.metric("Avg Win Rate",    f"{avg_wr:.1%}")
k4.metric("Avg PnL — Fear",  f"${fear_pnl:,.0f}",  delta=f"{'▲' if fear_pnl>0 else '▼'} Fear days")
k5.metric("Avg PnL — Greed", f"${greed_pnl:,.0f}", delta=f"{'▲' if greed_pnl>0 else '▼'} Greed days")

st.markdown("---")

# ── TAB LAYOUT ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🧠 Behavior", "👥 Segments", "💡 Insights"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">PnL & Win Rate by Sentiment</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = dark_fig((7,4))
        for sent in filtered['sentiment'].unique():
            sub = filtered[filtered['sentiment']==sent]['daily_pnl'].clip(-50000,50000)
            ax.hist(sub, bins=60, alpha=0.6, label=sent, color=sent_color(sent))
        ax.set_title('PnL Distribution by Sentiment', color='#c8c8d8', fontsize=11)
        ax.set_xlabel('Daily PnL (USD)')
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax.legend(facecolor=CARD_BG, labelcolor='white', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = dark_fig((7,4))
        sents   = filtered['sentiment'].unique()
        wr_data = [filtered[filtered['sentiment']==s]['win_rate'].dropna() for s in sents]
        colors  = [sent_color(s) for s in sents]
        bp = ax.boxplot(wr_data, labels=sents, patch_artist=True, medianprops={'color':'white','linewidth':2})
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.set_title('Win Rate Distribution', color='#c8c8d8', fontsize=11)
        ax.set_ylabel('Win Rate')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Time series
    st.markdown('<div class="section-header">Daily PnL Over Time</div>', unsafe_allow_html=True)
    daily_total = filtered.groupby(['date','sentiment'])['daily_pnl'].sum().reset_index().sort_values('date')
    fig, ax = dark_fig((14,4))
    for sent in daily_total['sentiment'].unique():
        sub = daily_total[daily_total['sentiment']==sent]
        ax.scatter(sub['date'], sub['daily_pnl'], color=sent_color(sent), alpha=0.5, s=15, label=sent)
    roll = daily_total.groupby('date')['daily_pnl'].sum().rolling(7,min_periods=1).mean()
    ax.plot(roll.index, roll.values, color=BLUE, linewidth=1.5, label='7-day MA', zorder=5)
    ax.axhline(0, color='#555', linestyle='--', linewidth=0.8)
    ax.set_title('Total Daily PnL — Colored by Sentiment', color='#c8c8d8', fontsize=11)
    ax.legend(facecolor=CARD_BG, labelcolor='white', fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Stat test
    st.markdown('<div class="section-header">Statistical Significance</div>', unsafe_allow_html=True)
    if len(fear_d) > 10 and len(greed_d) > 10:
        stat, p = stats.mannwhitneyu(fear_d['daily_pnl'].dropna(), greed_d['daily_pnl'].dropna(), alternative='two-sided')
        col1, col2, col3 = st.columns(3)
        col1.metric("Test",        "Mann-Whitney U")
        col2.metric("p-value",     f"{p:.4f}")
        col3.metric("Significant", "✅ YES" if p < 0.05 else "❌ NO")
        if p < 0.05:
            st.markdown('<div class="insight-box">✅ The difference in PnL between Fear and Greed days is <strong>statistically significant</strong> (p &lt; 0.05). This is not random — sentiment genuinely impacts trader performance.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">⚠️ The difference is <strong>not statistically significant</strong> at the 0.05 level. More data may be needed to confirm the pattern.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">How Traders Behave Under Different Sentiment</div>', unsafe_allow_html=True)

    behavior_metrics = {
        'trade_count': 'Trades per Day',
        'avg_size'   : 'Avg Position Size (USD)',
        'long_ratio' : 'Long Ratio (0=Short, 1=Long)',
    }
    behavior_metrics = {k:v for k,v in behavior_metrics.items() if k in filtered.columns}

    cols = st.columns(len(behavior_metrics))
    for col, (metric, title) in zip(cols, behavior_metrics.items()):
        fig, ax = dark_fig((5,4))
        avg = filtered.groupby('sentiment')[metric].mean()
        colors = [sent_color(s) for s in avg.index]
        bars = ax.bar(avg.index, avg.values, color=colors, alpha=0.8, edgecolor='#333', linewidth=0.8)
        ax.set_title(title, color='#c8c8d8', fontsize=10)
        ax.set_ylabel('Average')
        for bar, val in zip(bars, avg.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f'{val:.2f}', ha='center', va='bottom', color='#aaa', fontsize=9)
        col.pyplot(fig, use_container_width=True)
        plt.close()

    # Behavioral summary table
    st.markdown('<div class="section-header">Behavioral Summary Table</div>', unsafe_allow_html=True)
    behavior_cols = [c for c in ['trade_count','avg_size','win_rate','long_ratio'] if c in filtered.columns]
    summary_tbl = filtered.groupby('sentiment')[behavior_cols].mean().round(3)
    st.dataframe(summary_tbl.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Trader Segments</div>', unsafe_allow_html=True)
    seg_choice = st.selectbox("Segment by", ['perf_segment','freq_segment'])

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = dark_fig((6,4))
        grp = filtered.groupby(seg_choice)['win_rate'].mean()
        ax.bar(grp.index.astype(str), grp.values, color=BLUE, alpha=0.8, edgecolor='#333')
        ax.set_title('Win Rate by Segment', color='#c8c8d8', fontsize=11)
        ax.set_ylabel('Avg Win Rate')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = dark_fig((6,4))
        grp2 = filtered.groupby(seg_choice)['daily_pnl'].mean()
        colors = [GREED_COL if v >= 0 else FEAR_COL for v in grp2.values]
        ax.bar(grp2.index.astype(str), grp2.values, color=colors, alpha=0.8, edgecolor='#333')
        ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax.set_title('Avg Daily PnL by Segment', color='#c8c8d8', fontsize=11)
        ax.set_ylabel('Avg PnL (USD)')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Heatmap: segment × sentiment
    st.markdown('<div class="section-header">Segment × Sentiment Heatmap</div>', unsafe_allow_html=True)
    try:
        pivot = filtered.groupby([seg_choice,'sentiment'])['daily_pnl'].mean().unstack()
        fig, ax = plt.subplots(figsize=(8,4), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                    linewidths=0.5, ax=ax, annot_kws={'size':10,'color':'white'},
                    cbar_kws={'label':'Avg Daily PnL'})
        ax.set_title('Avg PnL: Trader Segment × Sentiment', color='#c8c8d8', fontsize=11)
        ax.tick_params(colors='#888')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"Heatmap unavailable: {e}")

    # Raw segment stats
    st.markdown('<div class="section-header">Segment Detail Table</div>', unsafe_allow_html=True)
    detail_cols = [c for c in ['daily_pnl','win_rate','trade_count','avg_size'] if c in filtered.columns]
    seg_detail = filtered.groupby([seg_choice,'sentiment'])[detail_cols].mean().round(3)
    st.dataframe(seg_detail.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — INSIGHTS & STRATEGIES
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

    fear_wr  = fear_d['win_rate'].mean()  if len(fear_d)  else 0
    greed_wr = greed_d['win_rate'].mean() if len(greed_d) else 0
    fear_tc  = fear_d['trade_count'].mean()  if len(fear_d)  else 0
    greed_tc = greed_d['trade_count'].mean() if len(greed_d) else 0

    st.markdown(f"""
    <div class="insight-box">
        <strong>📌 Insight 1 — PnL differs by sentiment</strong><br>
        Avg PnL on <span class="greed-badge">Greed</span> days: <strong>${greed_pnl:,.0f}</strong> vs
        <span class="fear-badge">Fear</span> days: <strong>${fear_pnl:,.0f}</strong>.
        {'Greed days are more profitable on average.' if greed_pnl > fear_pnl else 'Fear days show higher PnL — contrarian behavior may be at play.'}
    </div>
    <div class="insight-box">
        <strong>📌 Insight 2 — Win rate shifts with sentiment</strong><br>
        Win rate on <span class="greed-badge">Greed</span> days: <strong>{greed_wr:.1%}</strong> vs
        <span class="fear-badge">Fear</span> days: <strong>{fear_wr:.1%}</strong>.
        {'Traders win more often when sentiment is greedy.' if greed_wr > fear_wr else 'Traders surprisingly win more often on Fear days — possibly due to short-selling opportunities.'}
    </div>
    <div class="insight-box">
        <strong>📌 Insight 3 — Trade frequency changes with sentiment</strong><br>
        Avg trades/day on <span class="greed-badge">Greed</span>: <strong>{greed_tc:.1f}</strong> vs
        <span class="fear-badge">Fear</span>: <strong>{fear_tc:.1f}</strong>.
        {'Traders are more active on Greed days — FOMO-driven overtrading is a risk.' if greed_tc > fear_tc else 'Traders trade more on Fear days — panic trading may hurt performance.'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Strategy Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="strategy-box">
        <strong>🎯 Strategy 1 — Sentiment-Gated Leverage Rule</strong><br>
        On <span class="fear-badge">Fear</span> days → cap leverage at <strong>3x</strong> for all traders.<br>
        On <span class="greed-badge">Greed</span> days → allow up to <strong>5x</strong> for top-performing segment only.<br>
        <em>Rationale: Fear increases volatility — same leverage = higher actual risk exposure.</em>
    </div>
    <div class="strategy-box">
        <strong>🎯 Strategy 2 — Frequency Throttle on Greed Days</strong><br>
        Infrequent traders who suddenly increase activity on <span class="greed-badge">Greed</span> days show poor win rates.<br>
        Apply a <strong>max 3 trades/day</strong> guardrail for the Infrequent segment on Greed days.<br>
        <em>Rationale: FOMO-driven overtrading destroys edge — fewer, higher-conviction trades perform better.</em>
    </div>
    """, unsafe_allow_html=True)

    # Raw data explorer
    st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
    n_rows = st.slider("Rows to display", 10, 200, 50)
    st.dataframe(filtered.head(n_rows), use_container_width=True)
    st.download_button("⬇️ Download Filtered Data", filtered.to_csv(index=False),
                       file_name="filtered_daily_trader.csv", mime="text/csv")
