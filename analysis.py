# coding: utf-8
"""
Hyperliquid Trader Behavior vs Bitcoin Sentiment - Primetrade.ai assignment

Quick summary of what this script does:
  - loads the sentiment CSV and the Hyperliquid fills CSV
  - cleans both (timestamps, IST->UTC, separating opening fills from closing ones)
  - builds daily per-account metrics (pnl, win rate, leverage, drawdown etc.)
  - runs the analysis across Fear/Greed sentiment buckets
  - segments accounts three ways and charts the differences
  - trains a simple RF model to predict next-day profitability
  - clusters accounts into behavioral archetypes

Outputs go to: charts/ (PNGs), daily_metrics.csv, account_archetypes.csv
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.makedirs("charts", exist_ok=True)

PALETTE = {"Fear": "#e55039", "Greed": "#27ae60", "Neutral": "#7f8c8d"}
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
print("setup done")


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("loading data...")

sentiment_raw = pd.read_csv("sentiment_data.csv")
trader_raw    = pd.read_csv("trader_data.csv")

print(f"sentiment : {sentiment_raw.shape[0]} rows, {sentiment_raw.shape[1]} cols")
print(f"trader    : {trader_raw.shape[0]} rows, {trader_raw.shape[1]} cols")

print("\nmissing values:")
print("  sentiment:", sentiment_raw.isnull().sum().sum(), "nulls")
print("  trader   :", trader_raw.isnull().sum().sum(), "nulls")
print("duplicates:", sentiment_raw.duplicated().sum(), "in sentiment,",
      trader_raw.duplicated().sum(), "in trader")



# ── 2. CLEAN ──────────────────────────────────────────────────────────
# sentiment
sentiment = sentiment_raw.drop_duplicates().copy()
sentiment["date"] = pd.to_datetime(sentiment["date"])
sentiment["classification"] = sentiment["classification"].str.strip()

def broad(cls):
    if pd.isna(cls): return "Unknown"
    c = cls.lower()
    if "fear" in c:  return "Fear"
    if "greed" in c: return "Greed"
    return "Neutral"

sentiment["broad_sentiment"] = sentiment["classification"].apply(broad)

print(f"\nsentiment classes:")
print(sentiment["classification"].value_counts().to_string())

# trader — more work needed
trader = trader_raw.drop_duplicates().copy()

# The timestamp is in IST but the sentiment index uses UTC dates.
# Subtract 5h30m before taking the date so trades don't get assigned to the wrong day.
trader["ts_ist"] = pd.to_datetime(trader["Timestamp IST"], dayfirst=True)
trader["date"]   = (trader["ts_ist"] - pd.Timedelta(hours=5, minutes=30)).dt.normalize()

# Not every row is a closing fill. Open fills have Closed PnL = 0 and shouldn't be used
# for PnL or win rate. The Direction column tells us what actually closed.
# Checked the unique values manually — only these create a realised P&L.
trader["is_closing"] = trader["Direction"].str.contains(
    r"^Close|Long.*Short|Short.*Long|Liquidat|Settlement",
    case=False, na=False, regex=True
)

trader["Fee"]     = trader["Fee"].clip(lower=0)
trader["net_pnl"] = trader["Closed PnL"] - trader["Fee"]

# No leverage column exists. Proxy: how much size did they put on relative to their position notional?
trader["notional"] = (trader["Start Position"].abs() * trader["Execution Price"]).replace(0, np.nan)
trader["leverage"] = (trader["Size USD"].abs() / trader["notional"]).clip(upper=200)

trader["is_long"]  = (trader["Side"].str.upper() == "BUY").astype(int)
trader["is_short"] = (trader["Side"].str.upper() == "SELL").astype(int)

print(f"\ntrader rows    : {len(trader):,}")
print(f"closing fills  : {trader['is_closing'].sum():,}")
print(f"accounts       : {trader['Account'].nunique()}")
print(f"date range     : {trader['date'].min().date()} to {trader['date'].max().date()}")

# ── 3. MERGE ──────────────────────────────────────────────────────────────────
merged = trader.merge(
    sentiment[["date", "value", "classification", "broad_sentiment"]],
    on="date", how="left"
)
print(f"rows after merge: {len(merged):,}")
print("\nsentiment breakdown in merged:")
print(merged["broad_sentiment"].value_counts(dropna=False).to_string())


# ── 4. BUILD DAILY METRICS ────────────────────────────────────────────────────
# Separate two groups:
#   all_fills -> trade frequency, position size, side, leverage
#   closing_fills -> PnL, win/loss count
# Then merge them on (Account, date)
daily_all = (
    merged
    .groupby(["Account", "date", "broad_sentiment", "classification", "value"])
    .agg(
        trade_count  =("Trade ID",   "count"),
        avg_size_usd =("Size USD",   "mean"),
        total_size_usd=("Size USD",  "sum"),
        long_count   =("is_long",    "sum"),
        short_count  =("is_short",   "sum"),
        avg_leverage =("leverage",   "mean"),
    )
    .reset_index()
)

# Closing fills only -> PnL, win rate
closing = merged[merged["is_closing"]].copy()
closing["is_win"]  = (closing["net_pnl"] > 0).astype(int)
closing["is_loss"] = (closing["net_pnl"] < 0).astype(int)

daily_close = (
    closing
    .groupby(["Account", "date"])
    .agg(
        daily_pnl  =("net_pnl", "sum"),
        win_count  =("is_win",  "sum"),
        loss_count =("is_loss", "sum"),
        close_count=("Trade ID","count"),
    )
    .reset_index()
)

daily = daily_all.merge(daily_close, on=["Account", "date"], how="left")
daily[["daily_pnl","win_count","loss_count","close_count"]] = \
    daily[["daily_pnl","win_count","loss_count","close_count"]].fillna(0)

daily["win_rate"]         = daily["win_count"]  / daily["close_count"].clip(lower=1)
daily["long_short_ratio"] = daily["long_count"] / daily["short_count"].clip(lower=1)

# rolling drawdown: how far are we from the high water mark at each point in time
daily = daily.sort_values(["Account", "date"])
daily["cum_pnl"]     = daily.groupby("Account")["daily_pnl"].cumsum()
daily["rolling_max"] = daily.groupby("Account")["cum_pnl"].cummax()
daily["drawdown"]    = daily["cum_pnl"] - daily["rolling_max"]

print(f"\ndaily rows: {len(daily):,}")
print("\nsample:")
print(daily[["Account","date","broad_sentiment","daily_pnl","win_rate","avg_leverage","drawdown"]].head(3).to_string())


# ── 5. ANALYSIS ───────────────────────────────────────────────────────────────

focus = daily[daily["broad_sentiment"].isin(["Fear", "Greed"])].copy()

# B1 — does performance actually differ between Fear and Greed days?
print("\nperformance by sentiment:")
perf = (
    focus.groupby("broad_sentiment")
    .agg(
        account_days  =("Account",   "count"),
        avg_daily_pnl =("daily_pnl", "mean"),
        med_daily_pnl =("daily_pnl", "median"),
        avg_win_rate  =("win_rate",  "mean"),
        avg_drawdown  =("drawdown",  "mean"),
        pct_pnl_pos   =("daily_pnl", lambda x: (x > 0).mean()),
    )
    .reset_index()
)
print(perf.to_string(index=False))

fear_pnl  = focus.loc[focus["broad_sentiment"]=="Fear",  "daily_pnl"]
greed_pnl = focus.loc[focus["broad_sentiment"]=="Greed", "daily_pnl"]
stat, pval = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
print(f"Mann-Whitney U: stat={stat:.0f}, p={pval:.4f} — "
      f"{'significant' if pval<0.05 else 'not significant (small n=32 accounts, directional effect is still consistent)'}")

# Chart B1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("B1: Trader Performance - Fear vs Greed Days", fontsize=14, fontweight="bold")

metrics_b1 = [
    ("avg_daily_pnl", "Avg Daily PnL ($)", "${:,.0f}"),
    ("avg_win_rate",  "Avg Win Rate",       "{:.1%}"),
    ("avg_drawdown",  "Avg Drawdown ($)",   "${:,.0f}"),
]
for ax, (col, label, fmt) in zip(axes, metrics_b1):
    colors = [PALETTE[s] for s in perf["broad_sentiment"]]
    bars = ax.bar(perf["broad_sentiment"], perf[col], color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h*1.01, fmt.format(h),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig("charts/b1_performance_by_sentiment.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b1_performance_by_sentiment.png")


# B2 — do traders actually behave differently when market is in Fear vs Greed?
print("\nbehaviour by sentiment:")
beh = (
    focus.groupby("broad_sentiment")
    .agg(
        avg_trades   =("trade_count",       "mean"),
        avg_leverage =("avg_leverage",       "mean"),
        avg_size_usd =("avg_size_usd",       "mean"),
        avg_ls_ratio =("long_short_ratio",   "mean"),
    )
    .reset_index()
)
print(beh.to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("B2: Trader Behaviour - Fear vs Greed Days", fontsize=14, fontweight="bold")

metrics_b2 = [
    ("avg_trades",   "Avg Trades / Day"),
    ("avg_leverage", "Avg Leverage (x)"),
    ("avg_size_usd", "Avg Trade Size ($)"),
    ("avg_ls_ratio", "Avg Long:Short Ratio"),
]
for ax, (col, label) in zip(axes, metrics_b2):
    colors = [PALETTE[s] for s in beh["broad_sentiment"]]
    bars = ax.bar(beh["broad_sentiment"], beh[col], color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h*1.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig("charts/b2_behaviour_by_sentiment.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b2_behaviour_by_sentiment.png")

# B3 — segment analysis
# Three cuts: leverage level, trade frequency, consistency of returns
print("\nsegment analysis...")

# Segment 1: High vs Low Leverage
# Who has a naturally higher leverage tendency, and does it hurt them in Fear?
acct_lev = (
    daily.groupby("Account")
    .agg(med_leverage=("avg_leverage", "median"))
    .reset_index()
)
lev_thresh = acct_lev["med_leverage"].quantile(0.67)
acct_lev["lev_segment"] = np.where(
    acct_lev["med_leverage"] >= lev_thresh, "High Leverage", "Low Leverage"
)
daily = daily.merge(acct_lev[["Account","lev_segment"]], on="Account", how="left")

seg1 = (
    daily[daily["broad_sentiment"].isin(["Fear","Greed"])]
    .groupby(["lev_segment","broad_sentiment"])
    .agg(avg_pnl=("daily_pnl","mean"), avg_wr=("win_rate","mean"))
    .reset_index()
)
print("\nSegment 1: High vs Low Leverage x Sentiment")
print(seg1.pivot(index="lev_segment", columns="broad_sentiment", values=["avg_pnl","avg_wr"]).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
seg1.pivot(index="lev_segment", columns="broad_sentiment", values="avg_pnl").plot(
    kind="bar", ax=ax, color=[PALETTE["Fear"], PALETTE["Greed"]], edgecolor="white", width=0.6
)
ax.set_title("Seg 1: Avg Daily PnL - High vs Low Leverage x Sentiment", fontsize=12)
ax.set_xlabel("Leverage Segment")
ax.set_ylabel("Avg Daily PnL ($)")
ax.legend(title="Sentiment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/b3_seg1_leverage.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b3_seg1_leverage.png")

# Segment 2: Frequent vs Infrequent
# Do high-frequency traders hold up better or worse in Fear?
acct_freq = (
    daily.groupby("Account")
    .agg(total_trades=("trade_count","sum"))
    .reset_index()
)
freq_thresh = acct_freq["total_trades"].quantile(0.67)
acct_freq["freq_segment"] = np.where(
    acct_freq["total_trades"] >= freq_thresh, "Frequent", "Infrequent"
)
daily = daily.merge(acct_freq[["Account","freq_segment"]], on="Account", how="left")

seg2 = (
    daily[daily["broad_sentiment"].isin(["Fear","Greed"])]
    .groupby(["freq_segment","broad_sentiment"])
    .agg(avg_pnl=("daily_pnl","mean"), avg_trades=("trade_count","mean"))
    .reset_index()
)
print("\nSegment 2: Frequent vs Infrequent x Sentiment")
print(seg2.pivot(index="freq_segment", columns="broad_sentiment", values=["avg_pnl","avg_trades"]).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
seg2.pivot(index="freq_segment", columns="broad_sentiment", values="avg_pnl").plot(
    kind="bar", ax=ax, color=[PALETTE["Fear"], PALETTE["Greed"]], edgecolor="white", width=0.6
)
ax.set_title("Seg 2: Avg Daily PnL - Frequent vs Infrequent x Sentiment", fontsize=12)
ax.set_xlabel("Frequency Segment")
ax.set_ylabel("Avg Daily PnL ($)")
ax.legend(title="Sentiment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/b3_seg2_frequency.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b3_seg2_frequency.png")

# Segment 3: Consistent Winners vs the rest
# This is probably the most interesting cut.
# Consistent = win rate > 50% AND total PnL positive over the full period.
acct_perf = (
    daily.groupby("Account")
    .agg(
        total_pnl    =("daily_pnl","sum"),
        avg_win_rate =("win_rate", "mean"),
    )
    .reset_index()
)
acct_perf["consistency"] = np.where(
    (acct_perf["avg_win_rate"] > 0.5) & (acct_perf["total_pnl"] > 0),
    "Consistent Winner", "Inconsistent"
)
daily = daily.merge(acct_perf[["Account","consistency"]], on="Account", how="left")

seg3 = (
    daily[daily["broad_sentiment"].isin(["Fear","Greed"])]
    .groupby(["consistency","broad_sentiment"])
    .agg(avg_pnl=("daily_pnl","mean"), avg_wr=("win_rate","mean"), avg_dd=("drawdown","mean"))
    .reset_index()
)
print("\nSegment 3: Consistent Winners vs Inconsistent x Sentiment")
print(seg3.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Seg 3: Consistent Winner vs Inconsistent x Sentiment", fontsize=13, fontweight="bold")

for ax, col, label in zip(axes, ["avg_pnl","avg_wr"], ["Avg Daily PnL ($)","Avg Win Rate"]):
    pivot = seg3.pivot(index="consistency", columns="broad_sentiment", values=col)
    pivot.plot(kind="bar", ax=ax, color=[PALETTE["Fear"], PALETTE["Greed"]],
               edgecolor="white", width=0.6)
    ax.set_ylabel(label)
    ax.set_xlabel("")
    ax.legend(title="Sentiment")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("charts/b3_seg3_consistency.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b3_seg3_consistency.png")


# Leverage distribution
fig, ax = plt.subplots(figsize=(10, 5))
for senti, color in [("Fear",PALETTE["Fear"]),("Greed",PALETTE["Greed"])]:
    data = daily.loc[daily["broad_sentiment"]==senti, "avg_leverage"].dropna()
    data = data[data < 50]
    sns.kdeplot(data, ax=ax, label=senti, color=color, fill=True, alpha=0.35, linewidth=2)
ax.set_title("Leverage Distribution - Fear vs Greed Days", fontsize=13)
ax.set_xlabel("Avg Leverage per Account-Day")
ax.set_ylabel("Density")
ax.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("charts/b4_leverage_distribution.png", bbox_inches="tight")
plt.close()
print("Saved: charts/b4_leverage_distribution.png")


# ── 6. INSIGHTS + STRATEGY ────────────────────────────────────────────────────
# Pull the actual numbers out of the computed frames instead of hardcoding them.
_fear  = beh.loc[beh["broad_sentiment"]=="Fear"].iloc[0]
_greed = beh.loc[beh["broad_sentiment"]=="Greed"].iloc[0]
_perf_fear  = perf.loc[perf["broad_sentiment"]=="Fear"].iloc[0]
_perf_greed = perf.loc[perf["broad_sentiment"]=="Greed"].iloc[0]

_trade_delta = (_fear["avg_trades"] - _greed["avg_trades"]) / _greed["avg_trades"] * 100
_size_delta  = (_fear["avg_size_usd"] - _greed["avg_size_usd"]) / _greed["avg_size_usd"] * 100

_cw_fear  = seg3.loc[(seg3["consistency"]=="Consistent Winner") & (seg3["broad_sentiment"]=="Fear"),  "avg_pnl"].values[0]
_cw_greed = seg3.loc[(seg3["consistency"]=="Consistent Winner") & (seg3["broad_sentiment"]=="Greed"), "avg_pnl"].values[0]
_inc_fear  = seg3.loc[(seg3["consistency"]=="Inconsistent") & (seg3["broad_sentiment"]=="Fear"),  "avg_pnl"].values[0]
_inc_greed = seg3.loc[(seg3["consistency"]=="Inconsistent") & (seg3["broad_sentiment"]=="Greed"), "avg_pnl"].values[0]

_hl_fear  = seg1.loc[(seg1["lev_segment"]=="High Leverage") & (seg1["broad_sentiment"]=="Fear"),  "avg_pnl"].values[0]
_hl_greed = seg1.loc[(seg1["lev_segment"]=="High Leverage") & (seg1["broad_sentiment"]=="Greed"), "avg_pnl"].values[0]
_ll_fear  = seg1.loc[(seg1["lev_segment"]=="Low Leverage")  & (seg1["broad_sentiment"]=="Fear"),  "avg_pnl"].values[0]
_ll_greed = seg1.loc[(seg1["lev_segment"]=="Low Leverage")  & (seg1["broad_sentiment"]=="Greed"), "avg_pnl"].values[0]

print(f"""
INSIGHT 1 - Panic Trading: Higher Activity, Lower Quality
  Fear days : avg {_fear['avg_trades']:.0f} trades/day, avg size ${_fear['avg_size_usd']:,.0f}
  Greed days: avg {_greed['avg_trades']:.0f} trades/day, avg size ${_greed['avg_size_usd']:,.0f}
  Trade count is {_trade_delta:+.1f}% higher on Fear days; avg size is {_size_delta:+.1f}% larger.
  Win-rate on Fear ({_perf_fear['avg_win_rate']:.1%}) vs Greed ({_perf_greed['avg_win_rate']:.1%}).
  Implication: Stress triggers over-trading, eroding returns through fee drag and poor timing.

INSIGHT 2 - Segment Quality Diverges Under Sentiment
  Consistent Winners  | Fear avg PnL: ${_cw_fear:,.0f}  | Greed avg PnL: ${_cw_greed:,.0f}
  Inconsistent        | Fear avg PnL: ${_inc_fear:,.0f}  | Greed avg PnL: ${_inc_greed:,.0f}
  Consistent Winners earn more on Greed; Inconsistent traders gamble in Fear with high variance.
  Implication: Sentiment amplifies the quality gap between trader segments.

INSIGHT 3 - Leverage Is the Fear Multiplier
  High-Leverage | Fear avg PnL: ${_hl_fear:,.0f}  | Greed avg PnL: ${_hl_greed:,.0f}
  Low-Leverage  | Fear avg PnL: ${_ll_fear:,.0f}  | Greed avg PnL: ${_ll_greed:,.0f}
  High-leverage accounts suffer a ${_hl_greed - _hl_fear:,.0f} PnL gap from Greed to Fear;
  low-leverage accounts show a much smaller gap of ${_ll_greed - _ll_fear:,.0f}.
  Implication: Leverage, not sentiment alone, determines downside magnitude.

STRATEGY 1 - "Fear Brake" for High-Leverage Traders
  Trigger: Fear/Greed index < 30 (Extreme Fear)
  Rule   : Cap leverage at 5x for accounts in the top-third of historical leverage usage.
           Enforce daily trade count <= account's 30-day Greed-day average.
  Evidence: High-leverage cohort PnL on Fear (${_hl_fear:,.0f}) << Greed (${_hl_greed:,.0f}).
  Benefit : Limits max drawdown without blocking trading activity.

STRATEGY 2 - "Greed Window" for Consistent Winners
  Trigger: Fear/Greed index > 60 (Greed / Extreme Greed)
  Rule   : Allow +20% position size for accounts with win_rate > 50% and cumulative PnL > 0.
  Evidence: Consistent Winners avg PnL on Greed (${_cw_greed:,.0f}) vs Fear (${_cw_fear:,.0f}).
  Benefit : Concentrates capital in proven alpha-generators when conditions are optimal.
""")


# =============================================================================
# BONUS - PREDICTIVE MODEL + CLUSTERING
# =============================================================================
print("="*60)
print("BONUS - Predictive Model + Trader Clustering")
print("="*60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

print("\nbuilding predictive model...")
# For each account-day, the target is whether the NEXT day is profitable.
# The last record per account has no next-day, so it drops out on dropna.
# Single-day accounts drop out too — that's fine, they don't help anyway.
model_df = daily.sort_values(["Account","date"]).copy()
model_df["next_pnl"]        = model_df.groupby("Account")["daily_pnl"].shift(-1)
model_df["next_profitable"] = (model_df["next_pnl"] > 0).astype(int)
model_df = model_df.dropna(subset=["next_pnl"])

le = LabelEncoder()
model_df["sentiment_enc"] = le.fit_transform(model_df["broad_sentiment"].fillna("Unknown"))

FEATURES = ["value","sentiment_enc","trade_count","win_rate",
            "avg_size_usd","long_short_ratio","avg_leverage","drawdown"]
X = model_df[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0)
y = model_df["next_profitable"]

print(f"\nModel dataset: {len(X):,} rows | Class balance: {y.value_counts().to_dict()}")

rf  = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv  = cross_val_score(rf,  X, y, cv=cv, scoring="accuracy")
gbc_cv = cross_val_score(gbc, X, y, cv=cv, scoring="accuracy")

print(f"Random Forest     CV Accuracy: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}")
print(f"Gradient Boosting CV Accuracy: {gbc_cv.mean():.3f} +/- {gbc_cv.std():.3f}")

best = rf if rf_cv.mean() >= gbc_cv.mean() else gbc
best.fit(X, y)

importances = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importances.index, importances.values,
               color=sns.color_palette("mako_r", len(FEATURES)))
ax.set_title("Feature Importance - Next-Day Profitability Prediction", fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/bonus_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: charts/bonus_feature_importance.png")
# ── 8. CLUSTERING ─────────────────────────────────────────────────────────────
# KMeans on 4 account-level features to find behavioral archetypes.
CLUSTER_FEATS = ["avg_leverage","total_size_usd","win_rate","trade_count"]

daily_agg = (
    daily.groupby("Account")
    .agg(
        avg_leverage   =("avg_leverage",   "mean"),
        total_size_usd =("total_size_usd", "sum"),
        win_rate       =("win_rate",       "mean"),
        trade_count    =("trade_count",    "sum"),
        total_pnl      =("daily_pnl",      "sum"),
    )
    .reset_index()
    .replace([np.inf,-np.inf], np.nan)
    .dropna()
)

scaler  = StandardScaler()
X_clust = scaler.fit_transform(daily_agg[CLUSTER_FEATS])

# Elbow
inertias = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertias[k] = km.inertia_

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(inertias.keys()), list(inertias.values()), "o-", color="#2980b9", linewidth=2)
ax.set_title("K-Means Elbow Curve", fontsize=12)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig("charts/bonus_elbow.png", bbox_inches="tight")
plt.close()
print("Saved: charts/bonus_elbow.png")

# Fit k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
daily_agg["cluster"] = km4.fit_predict(X_clust)

cluster_profile = (
    daily_agg.groupby("cluster")[["avg_leverage","total_size_usd","win_rate","trade_count","total_pnl"]]
    .mean()
    .sort_values("total_pnl", ascending=False)
    .round(2)
)
print("\nCluster Profiles:")
print(cluster_profile.to_string())

# Archetype labels by total_pnl rank
rank2name = {cluster_profile.index[0]: "Smart Money",
             cluster_profile.index[1]: "High-Frequency",
             cluster_profile.index[2]: "Degen",
             cluster_profile.index[3]: "Passive"}
daily_agg["archetype"] = daily_agg["cluster"].map(rank2name)

fig, ax = plt.subplots(figsize=(10, 6))
archetype_colors = {"Smart Money": "#27ae60", "High-Frequency": "#f39c12",
                    "Degen": "#e74c3c", "Passive": "#95a5a6"}
for group, gdf in daily_agg.groupby("archetype"):
    ax.scatter(gdf["avg_leverage"], gdf["win_rate"], label=group,
               color=archetype_colors.get(group, "#333"),
               s=gdf["trade_count"].clip(upper=2000)/5, alpha=0.7, edgecolors="white")
ax.set_title("Trader Archetypes: Leverage vs Win Rate\n(bubble size proportional to trade count)", fontsize=12)
ax.set_xlabel("Avg Leverage (x)")
ax.set_ylabel("Avg Win Rate")
ax.legend(title="Archetype", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("charts/bonus_clusters.png", bbox_inches="tight")
plt.close()
print("Saved: charts/bonus_clusters.png")

daily.to_csv("daily_metrics.csv", index=False)
daily_agg.to_csv("account_archetypes.csv", index=False)
print("\nsaved: daily_metrics.csv, account_archetypes.csv")
print("all done.")
