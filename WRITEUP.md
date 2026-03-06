# 📝 Analysis Write-Up
**Trader Performance vs Market Sentiment — Primetrade.ai Assignment**

---

## Methodology

The analysis merges two datasets: Bitcoin Fear/Greed Index (daily sentiment scores) and Hyperliquid historical trader data (tick-level trades). 

**Data Cleaning:** Removed duplicates, filled missing PnL values with 0 (representing open trades), and standardized column names. The numeric Fear/Greed score (0–100) was binned into labels — Fear (0–44), Neutral (45–55), Greed (56–100) — then simplified to a binary Fear/Greed classification for cleaner analysis.

**Alignment:** Hyperliquid timestamps (Unix milliseconds) were floored to date level and left-joined with the daily sentiment table on the date key.

**Feature Engineering:** Per-trader daily metrics were computed — daily PnL, win rate (% trades with PnL > 0), average position size, trade frequency, and long/short ratio. Traders were segmented into three tiers (Winner / Neutral / Loser and Frequent / Moderate / Infrequent) using 33rd/67th percentile splits.

**Statistical Validation:** A Mann-Whitney U test (non-parametric, robust to skewed PnL distributions) was used to confirm whether PnL differences across sentiment groups are statistically significant (p < 0.05).

**Predictive Model:** A Random Forest classifier was trained to predict next-day trader profitability (binary: profit vs loss) using sentiment, trade frequency, position size, and long ratio as features. Performance was evaluated using 5-fold cross-validated ROC-AUC.

---

## Key Insights

**Insight 1 — Greed days produce higher PnL and win rates**
Average daily PnL and win rates are measurably higher on Greed days compared to Fear days. This difference is statistically significant (Mann-Whitney U, p < 0.05), confirming that sentiment is not just correlated with performance but is a reliable signal.

**Insight 2 — Frequent traders overtrade on Fear days**
High-frequency traders increase their trade count on Fear days but see a sharp drop in win rate. More trades in a volatile, fear-driven environment does not translate to more profit — it amplifies losses through poor entries and wider spreads.

**Insight 3 — Consistent Winners are sentiment-resilient**
Top-performing traders (Consistent Winners segment) maintain positive PnL even on Fear days, while the bottom segment (Consistent Losers) suffers disproportionately. This suggests skilled traders adapt their strategy to sentiment conditions, while weaker traders do not.

---

## Strategy Recommendations

**Strategy 1 — Sentiment-Gated Leverage Rule**
> *"On Fear days, cap leverage at 3x for all trader segments. On Greed days, allow up to 5x only for the Consistent Winner segment."*

Fear days increase realized volatility, meaning the same leverage carries significantly higher risk of liquidation. Capping leverage on Fear days directly reduces max drawdown without limiting upside on calmer days. Restricting the leverage increase to top performers on Greed days prevents over-leveraging by less skilled traders chasing momentum.

**Strategy 2 — Frequency Throttle for Infrequent Traders on Greed Days**
> *"Apply a maximum of 3 trades per day for Infrequent traders on Greed days."*

Infrequent traders who suddenly increase activity on Greed days show significantly lower win rates — a classic FOMO (Fear Of Missing Out) pattern. Forcing selectivity by capping daily trade count pushes this segment toward higher-conviction entries, improving their win rate and overall PnL.

---

*Analysis conducted using Python (pandas, scipy, scikit-learn, matplotlib, seaborn, streamlit). Full notebook and interactive dashboard available in the repository.*
