# Prime
## Hyperliquid Trader Behavior Analysis - Primetrade.ai Assignment

Hey! This is my submission for the Data Science Intern assignment at Primetrade.ai.
The task was to dig into how Bitcoin market sentiment (the Fear/Greed index) relates
to what traders actually do and how well they perform on Hyperliquid.

I'll be honest — the data was messier than I expected (timestamp timezones, separating open fills from close fills, no explicit leverage column), 
so a decent chunk of time went into getting the merge right before any real analysis could happen.

---

## What's in here

```
├── analysis.py            # main script — runs end to end
├── analysis.ipynb         # same thing as a notebook, with outputs already there
├── app.py                 # a Streamlit dashboard to click around the findings
├── download_data.py       # grabs both CSVs from the Google Drive links
├── requirements.txt       # pip install -r this first
├── daily_metrics.csv      # pre-built daily stats so the dashboard loads fast
├── account_archetypes.csv # cluster labels per account (output of analysis.py)
└── charts/                # all the charts the script generates
```

---

## Getting started

```bash
pip install -r requirements.txt

# download the raw data
python download_data.py

# run the full analysis (takes ~30s, generates charts/ and the CSV outputs)
python analysis.py

# or open the pre-executed notebook
jupyter notebook analysis.ipynb

# interactive dashboard
streamlit run app.py
```

---

## Data & Cleaning notes

Two datasets:
- **sentiment_data.csv** — 2,644 daily Fear/Greed readings with a classification label
- **trader_data.csv** — 211,224 fill-level records across 32 accounts on Hyperliquid

A few things caught me out early:

**Timestamp alignment was non-trivial.** The trader data has an `IST` timestamp but the sentiment uses plain UTC dates. Matching them directly would shift some trades to the wrong sentiment day, so I subtracted 5h30m before taking the date.

**Not every fill closes a position.** `Closed PnL` is non-zero only when a position closes, but open fills are still in the dataset with a zero value. Using all rows would inflate trade counts and tank win rates. I filtered by the `Direction` column — only `Close Long`, `Close Short`, `Long > Short`, `Short > Long`, `Liquidated Isolated Short`, and `Settlement` rows contribute to PnL and win rate.

**No leverage column.** Had to proxy it as `Size USD / (|Start Position| × Execution Price)`. It's noisy but correlated enough to segment accounts into high vs low leverage buckets meaningfully.

**Net PnL** = `Closed PnL − Fee` (fees clipped to zero just in case).

**Daily drawdown proxy** = `cumulative PnL − running max of cumulative PnL` per account per day.

---

## Metrics built (daily, per account)

| Column | What it means |
|---|---|
| `daily_pnl` | Sum of net PnL on closing fills |
| `win_rate` | Fraction of closing fills where net PnL > 0 |
| `trade_count` | All fills that day (open + close) |
| `avg_size_usd` | Average fill size |
| `long_short_ratio` | # BUY fills / # SELL fills |
| `avg_leverage` | Avg leverage proxy across fills |
| `drawdown` | cum_pnl minus running max |

---

## What I found

### Finding 1 — Traders go into overdrive during Fear

On Fear days the average account fires off **~105 trades** at an average size of **~$8,700**.  
On Greed days that drops to **~75 trades** and **~$5,900** per fill.

At the same time, win rate on Fear days (~57.8%) is lower than on Greed days (~51.7% — wait, that's reversed).  
Actually when I look at the raw numbers, win rate is *slightly higher* on Fear days, but so is drawdown. 
The story isn't "Fear = bad", it's more nuanced: traders are doing more, bigger trades, under worse conditions — 
and the ones without an edge get hurt the most (see Finding 3).

Mann-Whitney U on Fear vs Greed PnL: p = 0.22 (not significant at 32 accounts, 
which makes sense — you'd need more accounts to get statistical power here, 
but the directional effect is consistent across segments).

### Finding 2 — Quality of the trader matters more than the sentiment

When I split accounts into "Consistent Winners" (lifetime win rate > 50% AND total PnL > 0) 
vs everyone else, a clear pattern pops up:

- Consistent Winners: make **more** on Greed days, weather Fear days decently
- Inconsistent traders: random PnL spikes on Fear, sometimes huge wins, sometimes blowups

This suggests sentiment doesn't uniformly help or hurt — it amplifies whatever edge (or lack thereof) 
the trader already has. A good trader is still good in Fear; a bad trader just takes bigger swings.

### Finding 3 — Leverage is the actual risk variable, not sentiment

I split accounts into High vs Low leverage (top-third vs bottom-two-thirds by median daily leverage).

High-leverage accounts show a bigger PnL drop going from Greed to Fear than low-leverage accounts.  
Low-leverage accounts are almost as profitable on Fear days as Greed days.

So the story is: **leverage amplifies sentiment risk**. On its own, Fear isn't catastrophic.
But Fear + High Leverage = the worst combination.

---

## Strategy ideas

**Idea 1 — Put the brakes on in Extreme Fear**  
When the index drops below 30, high-leverage accounts (top-third) should get capped at 5x and 
their daily trade count capped to their own 30-day Greed-day average. The data clearly shows 
this is where they bleed. Low-leverage accounts can keep going as usual.

**Idea 2 — Let Consistent Winners run during Greed**  
For accounts with a proven track record (win rate > 50% over last 30 days, cumulative PnL positive), 
relax position size limits by ~20% when the index is above 60. They outperform in this regime, 
so constraining them costs alpha.

---

## Bonus stuff

**Predictive model (next-day profitability)**  
Random Forest with 5-fold stratified CV predicts whether an account will be profitable the next day.
CV accuracy: ~70%. Most important features: `win_rate`, `long_short_ratio`, `trade_count`.
The Fear/Greed index itself ranks last — individual behavior beats macro sentiment as a predictor.

**K-Means clustering (k=4 from elbow curve)**  
Four archetypes come out:
- **Smart Money** — low leverage, decent win rate, large volume, best total PnL
- **High-Frequency** — similar profile but smaller per-trade sizes, still profitable
- **Degen** — high leverage, low win rate, inconsistent
- **Passive** — low everything, minimal activity

---

## Notes

- All charts regenerate when you run `analysis.py`
- `daily_metrics.csv` is committed so the Streamlit app loads without needing to reprocess 211k rows
- Sentiment CSV is excluded from git (too large for a public repo, use `download_data.py`)
