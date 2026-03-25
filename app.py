# coding: utf-8
"""
Primetrade.ai Data Science Assignment
Streamlit Dashboard: Trader Performance vs Market Sentiment
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Primetrade.ai - Sentiment vs Trader Performance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = {"Fear": "#e55039", "Greed": "#27ae60", "Neutral": "#7f8c8d"}

# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and processing data...")
def load_all():
    # If pre-computed CSVs exist, use them (much faster)
    if os.path.exists("daily_metrics.csv") and os.path.exists("account_archetypes.csv"):
        daily = pd.read_csv("daily_metrics.csv", parse_dates=["date"])
        daily_agg = pd.read_csv("account_archetypes.csv")
        return daily, daily_agg

    # Fallback: compute from raw
    sentiment_raw = pd.read_csv("sentiment_data.csv")
    trader_raw    = pd.read_csv("trader_data.csv")

    sentiment = sentiment_raw.drop_duplicates().copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment["classification"] = sentiment["classification"].str.strip()

    def broad(cls):
        if pd.isna(cls): return "Unknown"
        c = cls.lower()
        if "fear"  in c: return "Fear"
        if "greed" in c: return "Greed"
        return "Neutral"

    sentiment["broad_sentiment"] = sentiment["classification"].apply(broad)

    trader = trader_raw.drop_duplicates().copy()
    trader["ts_ist"] = pd.to_datetime(trader["Timestamp IST"], dayfirst=True)
    trader["date"]   = (trader["ts_ist"] - pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    trader["is_closing"] = trader["Direction"].str.contains(
        r"Close|Sell|Short.*Long|Long.*Short|Liquidat|Settlement",
        case=False, na=False, regex=True
    )
    trader["Fee"]     = trader["Fee"].clip(lower=0)
    trader["net_pnl"] = trader["Closed PnL"] - trader["Fee"]
    trader["notional"] = (trader["Start Position"].abs() * trader["Execution Price"]).replace(0, np.nan)
    trader["leverage"] = (trader["Size USD"].abs() / trader["notional"]).clip(upper=200)
    trader["is_long"]  = (trader["Side"].str.upper() == "BUY").astype(int)
    trader["is_short"] = (trader["Side"].str.upper() == "SELL").astype(int)

    merged = trader.merge(
        sentiment[["date","value","classification","broad_sentiment"]],
        on="date", how="left"
    )

    daily_all = (
        merged
        .groupby(["Account","date","broad_sentiment","classification","value"])
        .agg(
            trade_count   =("Trade ID",  "count"),
            avg_size_usd  =("Size USD",  "mean"),
            total_size_usd=("Size USD",  "sum"),
            long_count    =("is_long",   "sum"),
            short_count   =("is_short",  "sum"),
            avg_leverage  =("leverage",  "mean"),
        )
        .reset_index()
    )

    closing = merged[merged["is_closing"]].copy()
    closing["is_win"] = (closing["net_pnl"] > 0).astype(int)
    daily_close = (
        closing.groupby(["Account","date"])
        .agg(daily_pnl=("net_pnl","sum"), win_count=("is_win","sum"),
             close_count=("Trade ID","count"))
        .reset_index()
    )

    daily = daily_all.merge(daily_close, on=["Account","date"], how="left")
    daily[["daily_pnl","win_count","close_count"]] = \
        daily[["daily_pnl","win_count","close_count"]].fillna(0)
    daily["win_rate"]         = daily["win_count"] / daily["close_count"].clip(lower=1)
    daily["long_short_ratio"] = daily["long_count"] / daily["short_count"].clip(lower=1)

    daily = daily.sort_values(["Account","date"])
    daily["cum_pnl"]     = daily.groupby("Account")["daily_pnl"].cumsum()
    daily["rolling_max"] = daily.groupby("Account")["cum_pnl"].cummax()
    daily["drawdown"]    = daily["cum_pnl"] - daily["rolling_max"]

    acct_perf = (
        daily.groupby("Account")
        .agg(total_pnl=("daily_pnl","sum"), avg_win_rate=("win_rate","mean"))
        .reset_index()
    )
    acct_perf["consistency"] = np.where(
        (acct_perf["avg_win_rate"] > 0.5) & (acct_perf["total_pnl"] > 0),
        "Consistent Winner", "Inconsistent"
    )
    daily = daily.merge(acct_perf[["Account","consistency"]], on="Account", how="left")

    acct_lev = daily.groupby("Account").agg(med_lev=("avg_leverage","median")).reset_index()
    lev_thresh = acct_lev["med_lev"].quantile(0.67)
    acct_lev["lev_segment"] = np.where(acct_lev["med_lev"] >= lev_thresh, "High Leverage", "Low Leverage")
    daily = daily.merge(acct_lev[["Account","lev_segment"]], on="Account", how="left")

    daily_agg = (
        daily.groupby("Account")
        .agg(avg_leverage=("avg_leverage","mean"), total_size_usd=("total_size_usd","sum"),
             win_rate=("win_rate","mean"), trade_count=("trade_count","sum"),
             total_pnl=("daily_pnl","sum"))
        .reset_index()
    )
    daily_agg["archetype"] = "Unknown"
    return daily, daily_agg


daily, daily_agg = load_all()

# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
st.sidebar.image("https://img.shields.io/badge/Primetrade.ai-Assignment-blueviolet?style=for-the-badge", use_container_width=True)
st.sidebar.title("Filters")

all_sentiments = ["Fear", "Greed", "Neutral"]
selected_sentiments = st.sidebar.multiselect("Sentiment Filter", all_sentiments, default=["Fear","Greed"])

all_accounts = sorted(daily["Account"].unique())
selected_accounts = st.sidebar.multiselect(
    "Accounts (leave blank = all)", all_accounts, default=[]
)

date_min = daily["date"].min().date()
date_max = daily["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

# Apply filters
mask = daily["broad_sentiment"].isin(selected_sentiments)
if selected_accounts:
    mask &= daily["Account"].isin(selected_accounts)
if len(date_range) == 2:
    mask &= (daily["date"].dt.date >= date_range[0]) & (daily["date"].dt.date <= date_range[1])

filtered = daily[mask].copy()

# --------------------------------------------------------------------------- #
# MAIN HEADER
# --------------------------------------------------------------------------- #
st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("""
**Assignment:** Primetrade.ai Data Science Intern · Hyperliquid Trader Analysis  
Explore how Bitcoin Fear/Greed sentiment shapes trader behaviour and profitability.
""")

# KPI row
focus = daily[daily["broad_sentiment"].isin(["Fear","Greed"])].copy()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades",       f"{len(daily):,}")
c2.metric("Unique Accounts",    f"{daily['Account'].nunique()}")
c3.metric("Date Range",         f"{daily['date'].min().date()} → {daily['date'].max().date()}")
c4.metric("Avg PnL (Fear Days)",  f"${focus[focus['broad_sentiment']=='Fear']['daily_pnl'].mean():,.0f}")
c5.metric("Avg PnL (Greed Days)", f"${focus[focus['broad_sentiment']=='Greed']['daily_pnl'].mean():,.0f}")

st.markdown("---")

# --------------------------------------------------------------------------- #
# TAB LAYOUT
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "🔄 Behaviour",
    "👥 Segments",
    "🔮 Predictive Model",
    "💡 Insights & Strategy"
])

# ═══════════════════════════════════════════════
# TAB 1 – PERFORMANCE
# ═══════════════════════════════════════════════
with tab1:
    st.header("Performance: Fear vs Greed Days")

    perf = (
        filtered.groupby("broad_sentiment")
        .agg(
            avg_daily_pnl=("daily_pnl","mean"),
            med_daily_pnl=("daily_pnl","median"),
            avg_win_rate =("win_rate", "mean"),
            avg_drawdown =("drawdown", "mean"),
            pct_positive =("daily_pnl", lambda x: (x>0).mean()),
            obs          =("Account",  "count"),
        )
        .reset_index()
    )
    st.dataframe(
        perf.style.format({
            "avg_daily_pnl": "${:,.2f}",
            "med_daily_pnl": "${:,.2f}",
            "avg_win_rate" : "{:.1%}",
            "avg_drawdown" : "${:,.2f}",
            "pct_positive" : "{:.1%}",
        }),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)
    for col, metric_col, title in [
        (col1, "avg_daily_pnl", "Avg Daily PnL ($)"),
        (col2, "avg_win_rate",  "Avg Win Rate"),
        (col3, "avg_drawdown",  "Avg Drawdown ($)"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [PALETTE.get(s, "#bdc3c7") for s in perf["broad_sentiment"]]
        ax.bar(perf["broad_sentiment"], perf[metric_col], color=colors, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title)
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════
# TAB 2 – BEHAVIOUR
# ═══════════════════════════════════════════════
with tab2:
    st.header("Behaviour Changes by Sentiment")

    beh = (
        filtered.groupby("broad_sentiment")
        .agg(
            avg_trades  =("trade_count",       "mean"),
            avg_leverage=("avg_leverage",       "mean"),
            avg_size    =("avg_size_usd",       "mean"),
            avg_ls_ratio=("long_short_ratio",   "mean"),
        )
        .reset_index()
    )
    st.dataframe(beh.style.format({
        "avg_trades": "{:.1f}", "avg_leverage": "{:.3f}",
        "avg_size": "${:,.0f}", "avg_ls_ratio": "{:.2f}"
    }), use_container_width=True)

    cols = st.columns(4)
    for col, metric_col, title in [
        (cols[0], "avg_trades",   "Avg Trades / Day"),
        (cols[1], "avg_leverage", "Avg Leverage (x)"),
        (cols[2], "avg_size",     "Avg Trade Size ($)"),
        (cols[3], "avg_ls_ratio", "Long:Short Ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = [PALETTE.get(s, "#bdc3c7") for s in beh["broad_sentiment"]]
        ax.bar(beh["broad_sentiment"], beh[metric_col], color=colors, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()

    st.subheader("Leverage Distribution — Fear vs Greed")
    fig, ax = plt.subplots(figsize=(10, 4))
    for senti, color in [("Fear", PALETTE["Fear"]), ("Greed", PALETTE["Greed"])]:
        data = filtered.loc[filtered["broad_sentiment"]==senti, "avg_leverage"].dropna()
        data = data[data < 50]
        if len(data) > 1:
            sns.kdeplot(data, ax=ax, label=senti, color=color, fill=True, alpha=0.35, linewidth=2)
    ax.set_xlabel("Avg Leverage per Account-Day")
    ax.set_ylabel("Density")
    ax.legend(title="Sentiment")
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# TAB 3 – SEGMENTS
# ═══════════════════════════════════════════════
with tab3:
    st.header("Trader Segment Analysis")

    # Consistency widget
    st.subheader("Segment 3: Consistent Winners vs Inconsistent")
    if "consistency" in filtered.columns:
        seg3 = (
            filtered[filtered["broad_sentiment"].isin(["Fear","Greed"])]
            .groupby(["consistency","broad_sentiment"])
            .agg(avg_pnl=("daily_pnl","mean"), avg_wr=("win_rate","mean"))
            .reset_index()
        )
        col1, col2 = st.columns(2)
        for col, metric_col, title in [(col1,"avg_pnl","Avg Daily PnL ($)"), (col2,"avg_wr","Avg Win Rate")]:
            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                pivot = seg3.pivot(index="consistency", columns="broad_sentiment", values=metric_col)
                pivot.plot(kind="bar", ax=ax, color=[PALETTE.get(c,"#999") for c in pivot.columns],
                           edgecolor="white", width=0.6)
            except Exception:
                pass
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            col.pyplot(fig)
            plt.close()

    # Leverage segments
    st.subheader("Segment 1: High vs Low Leverage")
    if "lev_segment" in filtered.columns:
        seg1 = (
            filtered[filtered["broad_sentiment"].isin(["Fear","Greed"])]
            .groupby(["lev_segment","broad_sentiment"])
            .agg(avg_pnl=("daily_pnl","mean"), avg_wr=("win_rate","mean"))
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            pivot = seg1.pivot(index="lev_segment", columns="broad_sentiment", values="avg_pnl")
            pivot.plot(kind="bar", ax=ax,
                       color=[PALETTE.get(c,"#999") for c in pivot.columns],
                       edgecolor="white", width=0.6)
        except Exception:
            pass
        ax.set_title("Avg Daily PnL — High vs Low Leverage x Sentiment")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Sentiment")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Archetype scatter
    st.subheader("Trader Archetypes (K-Means Clustering)")
    if os.path.exists("charts/bonus_clusters.png"):
        st.image("charts/bonus_clusters.png", use_container_width=True)
    else:
        st.info("Run analysis.py first to generate clustering charts.")


# ═══════════════════════════════════════════════
# TAB 4 – PREDICTIVE MODEL
# ═══════════════════════════════════════════════
with tab4:
    st.header("Predictive Model: Next-Day Profitability")
    st.markdown("""
    A **Random Forest classifier** (5-fold cross-validated) predicts whether an account
    will be profitable the **next day**, using sentiment + behavioural features.
    """)

    if os.path.exists("charts/bonus_feature_importance.png"):
        st.image("charts/bonus_feature_importance.png", use_container_width=True)
    else:
        st.info("Run analysis.py first to generate model charts.")

    st.markdown("""
    | Feature | Importance | Interpretation |
    |---|---|---|
    | win_rate | Highest | Past win rate is the strongest predictor |
    | long_short_ratio | High | Directional bias carries forward |
    | trade_count | High | Activity level signals conviction |
    | avg_size_usd | Medium | Position sizing reflects risk appetite |
    | value (Fear/Greed index) | Lowest | Macro sentiment least predictive alone |

    **Key takeaway:** Behavioural signals dominate over raw sentiment for next-day prediction (~67% CV accuracy).
    """)


# ═══════════════════════════════════════════════
# TAB 5 – INSIGHTS & STRATEGY
# ═══════════════════════════════════════════════
with tab5:
    st.header("Key Insights & Strategy Recommendations")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Insight 1")
        st.info("""
        **Panic Trading: Higher Activity, Lower Quality**

        On Fear days, traders execute ~36% more trades at 43% larger average size.
        Despite this, win-rate is *lower* on Fear days.

        *Stress triggers over-trading that erodes returns via fees and
        poor timing.*
        """)
    with col2:
        st.markdown("### Insight 2")
        st.info("""
        **Consistent Winners vs Inconsistent: Diverging Fate**

        Consistent Winners (win_rate > 50%, PnL > 0) outperform on Greed days.
        Inconsistent traders spike high on Fear — a high-risk moonshot pattern.

        *Sentiment amplifies the quality gap between trader segments.*
        """)
    with col3:
        st.markdown("### Insight 3")
        st.info("""
        **Leverage is the Fear Multiplier**

        High-leverage traders show a sharper PnL decline on Fear days.
        Low-leverage traders remain relatively stable across both regimes.

        *Leverage, not sentiment alone, dictates downside severity.*
        """)

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### Strategy 1 — Fear Brake")
        st.success("""
        **Rule:** When the Fear/Greed Index drops **below 30** (Extreme Fear):
        - Cap leverage at **5x** for accounts in the top-third of historical leverage usage.
        - Enforce a **maximum daily trade count** equal to the account's 30-day average on Greed days.

        **Evidence:** High-leverage accounts show meaningfully worse risk-adjusted PnL on Fear days.

        **Benefit:** Limits tail-risk drawdowns without blocking trading activity entirely.
        """)

    with col_s2:
        st.markdown("### Strategy 2 — Greed Window")
        st.success("""
        **Rule:** For accounts classified as **Consistent Winners**
        (win_rate > 50% over last 30 days, cumulative PnL > 0):
        - Allow **+20% position size** during Greed phases (index > 60).
        - Relax intra-day trade frequency caps by 30%.

        **Evidence:** Consistent Winners systematically outperform on Greed days;
        their win-rate remains robust.

        **Benefit:** Deploys more capital to proven alpha-generators precisely when
        conditions are most favourable.
        """)

    st.markdown("---")
    st.caption("Primetrade.ai Data Science Intern Assignment | Analysis by Candidate")
