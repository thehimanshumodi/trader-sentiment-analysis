# 📊 Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

> Analyzing how Bitcoin Fear/Greed sentiment relates to trader behavior and PnL on Hyperliquid.

---

## 📁 Project Structure

```
trader-sentiment-analysis/
│
├── trader_sentiment_analysis.ipynb   # Full analysis notebook
├── dashboard.py                      # Interactive Streamlit dashboard
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/trader-sentiment-analysis.git
cd trader-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> If you get a file permission error (common on Windows with Anaconda), use:
> ```bash
> pip install -r requirements.txt --user
> ```

### 3. Add your data files
Download both datasets and place them in the project root:
- `fear_greed.csv` — Bitcoin Fear & Greed Index
- `trades.csv` — Hyperliquid historical trader data

> ⚠️ Data files are excluded from the repo via `.gitignore`

---

## 📓 Running the Notebook

```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

Run all cells top to bottom. The notebook will generate 6 charts automatically:

| Chart | Description |
|-------|-------------|
| `chart1_pnl_by_sentiment.png` | PnL distribution, win rate, avg PnL — Fear vs Greed |
| `chart2_behavior_by_sentiment.png` | Trade count, size, leverage, long ratio by sentiment |
| `chart3_segments.png` | Win rate & PnL across trader segments |
| `chart4_heatmap.png` | Heatmap: segment × sentiment PnL matrix |
| `chart5_timeseries.png` | Daily PnL over time, colored by sentiment |
| `chart6_model.png` | Feature importance + confusion matrix |

---

## 🖥️ Running the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

### Dashboard Features

| Tab | Contents |
|-----|----------|
| 📈 **Performance** | PnL distribution, win rate boxplot, time-series chart, Mann-Whitney significance test |
| 🧠 **Behavior** | Trade count, position size, long ratio charts + summary table |
| 👥 **Segments** | Winner/Loser/Neutral breakdown, segment × sentiment heatmap |
| 💡 **Insights** | Auto-generated insights + strategy recommendations |

### How to use the dashboard
1. Run `streamlit run dashboard.py`
2. Upload `trades.csv` and `fear_greed.csv` using the sidebar file uploaders
3. Use the **Sentiment** and **Date Range** filters to explore subsets
4. Download filtered data using the export button in the Insights tab

---

## 🔍 Methodology

1. **Data Cleaning** — Removed duplicates, handled missing PnL, fixed timestamp types
2. **Alignment** — Converted Unix-ms timestamps to dates, merged on daily date key
3. **Feature Engineering** — Win rate, leverage, long/short ratio, trade frequency
4. **Segmentation** — Traders split into frequency / performance tiers using percentiles
5. **Statistical Testing** — Mann-Whitney U test to confirm sentiment impact on PnL
6. **Predictive Model** — Random Forest classifier for next-day profitability (ROC-AUC ~0.65–0.70)

---

## 💡 Key Insights

- **Greed days outperform Fear days** in both PnL and win rate (statistically significant)
- **Frequent traders overtrade on Fear days**, reducing their edge in volatile conditions
- **Consistent Winners are sentiment-resilient**; Consistent Losers are disproportionately harmed by Fear

## 🎯 Strategy Recommendations

1. **Sentiment-Gated Leverage Rule** — Cap leverage at 3x on Fear days; allow up to 5x for top traders on Greed days
2. **Frequency Throttle** — Limit infrequent traders to max 3 trades/day on Greed days to prevent FOMO overtrading

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data wrangling and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualizations |
| `scipy` | Statistical testing (Mann-Whitney U) |
| `scikit-learn` | Random Forest predictive model |
| `streamlit` | Interactive dashboard |

---

*Submitted by: Himanshu Modi 