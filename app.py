import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_utils
import forecasting
import inventory_analysis as inv
import ml_forecast
import simulation

st.set_page_config(page_title="Colonial Fresh — AI Demand Forecasting & Inventory", layout="wide")

# ---------------------------------------------------------------- load data
df = data_utils.load_data()          # the 70% TRAINING file (2023-01 -> 2025-04)
TEST_DF = data_utils.load_test_data()  # the separate 30% TEST file (2025-05 -> 2026-04)
STORES = data_utils.get_stores(df)
PRODUCTS_DF = data_utils.get_products(df)
PRODUCTS = PRODUCTS_DF["Product_ID"].tolist()
PRODUCT_NAME = dict(zip(PRODUCTS_DF["Product_ID"], PRODUCTS_DF["Product_Name"]))
HISTORICAL_DF = df
HISTORICAL_AS_OF_DATE = data_utils.get_as_of_date(df)

# ---------------------------------------------------------------- session state
if "sim_state" not in st.session_state:
    st.session_state.sim_state = simulation.init_simulation_state(
        HISTORICAL_DF, STORES, PRODUCTS, HISTORICAL_AS_OF_DATE, seed=7, actuals_df=TEST_DF
    )
if "sim_autoplay" not in st.session_state:
    st.session_state.sim_autoplay = False
if "sim_speed" not in st.session_state:
    st.session_state.sim_speed = 1
if "stock_ledger" not in st.session_state:
    st.session_state.stock_ledger = []
if "placed_orders" not in st.session_state:
    st.session_state.placed_orders = []

_pending_reset = st.session_state.pop("_intake_pending_reset", None)
if _pending_reset:
    for _k, _v in _pending_reset.items():
        st.session_state[_k] = _v

sim_state = st.session_state.sim_state
LIVE_MODE = sim_state["days_simulated"] > 0

if LIVE_MODE:
    df = simulation.extended_dataframe(HISTORICAL_DF, sim_state["log"], PRODUCTS)
    AS_OF_DATE = sim_state["sim_date"]
    safety_stock_rates = simulation.current_safety_stock_rates(sim_state)
    # Real pending-order pipeline (not a historical-average guess) — this is
    # what stops Live Mode from misreading stock that's already on its way
    # as a stockout risk.
    incoming_pipeline = simulation.current_incoming_pipeline(sim_state)
else:
    AS_OF_DATE = HISTORICAL_AS_OF_DATE
    safety_stock_rates = {}
    incoming_pipeline = None


def _stock_adjustments_dict():
    agg = {}
    for entry in st.session_state.stock_ledger:
        key = (entry["store"], entry["sku"])
        agg[key] = agg.get(key, 0) + entry["qty"]
    return agg


stock_adjustments = _stock_adjustments_dict()

# ---------------------------------------------------------------- header
st.title("🥑 Colonial Fresh — AI-Based Demand Forecasting & Inventory Management")
st.caption("MGT5ELS Assessment 3 · Sprint 4 — Model Comparison (Random Forest vs Gradient Boosting)")

st.info(
    "**Prototype notice:** This dashboard runs on a fully **simulated** dataset built for academic purposes. "
    f"It loads the **70% training file** ({HISTORICAL_DF['Date'].min().date()} to {HISTORICAL_AS_OF_DATE.date()}); "
    f"the separate **30% test file** ({TEST_DF['Date'].min().date()} to {TEST_DF['Date'].max().date()}) is held out "
    "and only used to test the model and drive the Live Simulation. "
    "It is **not** real Colonial Fresh data and is **not** connected to any live system. "
    f"The most recent date in the dataset ({HISTORICAL_AS_OF_DATE.date()}) is treated as **'today'** "
    "for all current-inventory, lead-time and expiry calculations, to demonstrate how a real-time "
    "dashboard would behave. Live system integration is proposed as future work beyond this Sprint design."
)

if LIVE_MODE:
    st.success(
        f"🔴 **LIVE MODE ACTIVE** — the simulation has run **{sim_state['days_simulated']} day(s)** "
        f"forward, to **{AS_OF_DATE.date()}**. Every tab below (Overview, Movement, Forecasting, "
        "Inventory, Lead-Time, Expiry, Reorder, Risk) now reads this extended, simulated timeline — "
        "including each combo's **adaptively learned safety-stock rate** instead of the fixed 20% "
        "baseline. Go to 🔴 **Live Simulation** to advance further or reset back to the real historical data only."
    )

with st.sidebar:
    st.header("Settings")
    forecast_period = st.selectbox("Forecast period (days)", [7, 14, 30], index=0)
    st.caption("Every tab below always shows **all 6 stores × all 3 SKUs** together — nothing is filtered or hidden.")

# ---------------------------------------------------------------- shared tables
risk_table = inv.build_risk_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period,
                                   stock_adjustments, safety_stock_rates, incoming_pipeline)
full_table = inv.build_full_snapshot_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period,
                                            stock_adjustments, safety_stock_rates, incoming_pipeline)

tabs = st.tabs([
    "1️⃣ Overview", "2️⃣ Movement Analysis", "3️⃣ Demand Forecasting", "🧠 ML Forecast",
    "4️⃣ Inventory Monitoring", "📥 Stock Intake", "📦 Manual Order", "5️⃣ Lead-Time Tracking", "🚚 Lead-Time Anticipation",
    "6️⃣ Expiry Tracking", "7️⃣ Order Management Dashboard", "8️⃣ Risk Dashboard",
    "🔴 Live Simulation", "9️⃣ Solution Architecture",
])
(tab_overview, tab_movement, tab_forecast, tab_ml, tab_inventory, tab_intake, tab_order,
 tab_leadtime, tab_leadtime_anticipation, tab_expiry, tab_reorder, tab_risk,
 tab_live, tab_arch) = tabs

# ============================================================== 1. OVERVIEW
with tab_overview:
    st.header("Overview Dashboard")
    st.caption(f"All 6 stores × all 3 SKUs · forecast period = {forecast_period} days · as of {AS_OF_DATE.date()}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Current Inventory (units)", f"{full_table['Current_Stock'].sum():,.0f}")
    c2.metric(f"Forecasted Demand (next {forecast_period}d)", f"{full_table['Forecast_Next_Period'].sum():,.0f}")
    today_incoming = df[df["Date"] == AS_OF_DATE]["Incoming_Stock"].sum()
    c3.metric("Incoming Stock (today)", f"{today_incoming:,.0f}")
    c4.metric("Wastage (last 30d)", f"{full_table['Wastage_Last30d'].sum():,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Stockout Occurrences (last 30d)", f"{full_table['Stockout_Last30d'].sum():,.0f}")
    n_attention = int((full_table["Risk"] != "Normal").sum())
    c6.metric("Store × SKU Combos Needing Attention", f"{n_attention} / {len(full_table)}")
    c7.metric("Total Recommended Reorder Qty", f"{full_table['Recommended_Order'].sum():,.0f} units")

    st.divider()
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.subheader("Products / Stores Requiring Attention")
        attention = full_table[full_table["Risk"] != "Normal"][
            ["Store", "SKU", "Product", "Risk", "Current_Stock", "Recommended_Order"]
        ].sort_values("Risk")
        st.dataframe(attention, use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Risk Mix (current scope)")
        risk_counts = full_table["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.5,
                     color="Risk", color_discrete_map={
                         "Normal": "#2ca02c", "Overstock Risk": "#ff7f0e",
                         "Stockout Risk": "#d62728", "Expiry Risk": "#9467bd",
                     })
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== 2. MOVEMENT ANALYSIS
with tab_movement:
    st.header("Product Movement Analysis")
    st.caption("Movement class is **calculated** from average daily `Units_Sold` across the full "
               f"{HISTORICAL_DF['Date'].min().year}-{HISTORICAL_AS_OF_DATE.year} history for each SKU — not assumed.")

    move_summary = HISTORICAL_DF.groupby(["Product_ID", "Product_Name"]).agg(
        Avg_Daily_Units_Sold=("Units_Sold", "mean"),
        Max_Daily_Units_Sold=("Units_Sold", "max"),
        Min_Daily_Units_Sold=("Units_Sold", "min"),
    ).reset_index()
    move_class = HISTORICAL_DF.groupby("Product_ID")["Movement_Class"].first()
    move_summary["Calculated_Movement_Class"] = move_summary["Product_ID"].map(move_class)
    move_summary = move_summary.sort_values("Avg_Daily_Units_Sold", ascending=False)
    st.dataframe(move_summary, use_container_width=True, hide_index=True)

    fig = px.bar(move_summary, x="Product_Name", y="Avg_Daily_Units_Sold",
                 color="Calculated_Movement_Class", text_auto=".1f",
                 title=f"Average Daily Units Sold by Product ({HISTORICAL_DF['Date'].min().year}-{HISTORICAL_AS_OF_DATE.year} full history)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================== 3. DEMAND FORECASTING (baseline)
with tab_forecast:
    st.header("Demand Forecasting — All Stores × All SKUs (Baseline)")
    st.markdown(
        "**Baseline model:** 7-day moving average of historical `Units_Sold`, projected forward as a "
        "flat daily rate. `Simulated_True_Demand` is **not** used as a model input, to avoid data "
        "leakage — it is reserved for later evaluation only."
    )
    network_daily = df.groupby(["Date", "Product_ID"])["Units_Sold"].sum().reset_index()
    fig = go.Figure()
    colors = {"SKU001": "green", "SKU002": "orange", "SKU003": "purple"}
    for sku in PRODUCTS:
        d = network_daily[network_daily["Product_ID"] == sku].sort_values("Date").tail(60)
        name = PRODUCT_NAME[sku]
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Units_Sold"], name=f"{name} — Historical",
                                  line=dict(color=colors[sku])))
        daily_series = d.set_index("Date")["Units_Sold"]
        avg = forecasting.baseline_moving_average_forecast(daily_series, forecast_period)
        future_dates = pd.date_range(AS_OF_DATE + pd.Timedelta(days=1), periods=forecast_period)
        fig.add_trace(go.Scatter(x=future_dates, y=[avg] * forecast_period,
                                  name=f"{name} — Forecast", line=dict(color=colors[sku], dash="dash")))
    fig.update_layout(title="Network-Wide Historical Sales vs Forecast (all stores combined, per SKU)",
                       xaxis_title="Date", yaxis_title="Units Sold (all 6 stores)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast summary — every store × SKU")
    st.dataframe(
        full_table[["Store", "Product", "SKU", "Avg_Daily_Forecast", "Forecast_Next_Period"]]
        .rename(columns={"Forecast_Next_Period": f"Forecast (next {forecast_period}d)"}),
        use_container_width=True, hide_index=True,
    )

# ============================================================== ML FORECAST
with tab_ml:
    st.header("Machine Learning Forecast — Random Forest vs Gradient Boosting (network-wide, per SKU)")
    st.markdown(
        "Two genuine machine-learning models — scikit-learn `RandomForestRegressor` and `GradientBoostingRegressor` — "
        "trained on the same lag, rolling-average and calendar features engineered from historical `Units_Sold`. "
        "`Simulated_True_Demand` is never used as a feature or target, for the same data-leakage reasons as the baseline model."
    )
    st.markdown(
        "**Per lecturer feedback:** both models **learn only from the 70% training file** "
        f"({HISTORICAL_DF['Date'].min().date()} → {HISTORICAL_AS_OF_DATE.date()}) and are then checked against the "
        f"**separate 30% file** ({TEST_DF['Date'].min().date()} → {TEST_DF['Date'].max().date()}) they were never trained on — "
        "chronological, never shuffled, so no future information leaks into training. This tab answers two questions: "
        "**(1) which model is better, and (2) can the better model be trusted by itself?**"
    )

    sku_choice = st.selectbox("SKU", PRODUCTS, format_func=lambda s: f"{s} — {PRODUCT_NAME[s]}")

    @st.cache_data(show_spinner="Training and backtesting Random Forest and Gradient Boosting…")
    def _cached_backtest(sku):
        return ml_forecast.train_and_backtest(HISTORICAL_DF, sku, TEST_DF)

    bundle = _cached_backtest(sku_choice)
    LBL = ml_forecast.MODEL_LABELS
    win = bundle["winner"]
    lose = "gb" if win == "rf" else "rf"

    c1, c2 = st.columns(2)
    c1.metric("Training days (70% file)", f"{bundle['train_days']}")
    c1.caption(f"{bundle['train_date_range'][0].date()} → {bundle['train_date_range'][1].date()}")
    c2.metric("Test days (30% file, never trained on)", f"{bundle['test_days']}")
    c2.caption(f"{bundle['test_date_range'][0].date()} → {bundle['test_date_range'][1].date()}")
    st.caption(
        f"Inside the 70%, the last {bundle['bias_val_days']} days are used only as a validation slice to choose each model's "
        "settings (never the 30% test file). Each final model is then refit on all of the 70%."
    )

    # ------------------------------------------------------------ 1. which model is better
    st.subheader("1️⃣ Which model is better? — compared on the 30% test file (MAE)")
    variants = [
        ("7-day moving average (baseline)", "baseline"),
        ("Random Forest — original", "rf_original"),
        ("Random Forest — tuned", "rf_tuned"),
        ("Random Forest — FINAL (+ bias correction)", "rf"),
        ("Gradient Boosting — tuned", "gb_tuned"),
        ("Gradient Boosting — FINAL (+ bias correction)", "gb"),
    ]
    just = pd.DataFrame([{
        "Forecast": label + ("  ⭐ winner" if k == win else ""),
        "MAE": f"{bundle[k + '_metrics']['MAE']:.2f}",
        "RMSE": f"{bundle[k + '_metrics']['RMSE']:.2f}",
        "MAPE %": f"{bundle[k + '_metrics']['MAPE']:.1f}",
        "Avg error / day": f"{bundle[k + '_mean_error']:+.2f}",
        "Days outside ±4": bundle[k + "_ts_breaches"],
    } for label, k in variants])
    st.dataframe(just, use_container_width=True, hide_index=True)
    st.caption("All rows scored only on the 30% test file. Avg error = actual − forecast: + means the forecast ran too low "
               f"(under-forecasting), − means too high (over-forecasting). Days outside ±4 are out of {bundle['test_days']}. "
               "'Tuned' = settings picked on the validation slice; FINAL adds the adaptive bias correction. The winner is the FINAL model with the lowest MAE.")

    base_m, w_m, l_m = bundle["baseline_metrics"], bundle[win + "_metrics"], bundle[lose + "_metrics"]
    mae_gain = (base_m["MAE"] - w_m["MAE"]) / base_m["MAE"] * 100
    rmse_gain = (base_m["RMSE"] - w_m["RMSE"]) / base_m["RMSE"] * 100
    gap = abs(w_m["MAE"] - l_m["MAE"]) / l_m["MAE"] * 100
    ts_w, ts_l = bundle[win + "_ts_breaches"], bundle[lose + "_ts_breaches"]
    ts_note = (f" On the tracking signal, {LBL[win]} spends {ts_w} days outside ±4 against {ts_l} for {LBL[lose]}."
               if ts_w != ts_l else "")
    close_note = (
        f" The two models are very close (MAE differs by only {gap:.1f}%), so both are sound choices; "
        f"{LBL[win]} is simply the one with the smaller error here."
        if gap < 3 else
        f" {LBL[win]} is clearly ahead of {LBL[lose]} ({gap:.0f}% lower MAE)."
    ) + ts_note
    if w_m["MAE"] < base_m["MAE"] and w_m["RMSE"] < base_m["RMSE"]:
        st.success(
            f"**Winner for {sku_choice}: {LBL[win]}.** Lowest MAE on the 30% test file: {w_m['MAE']:.2f} units/day against "
            f"{l_m['MAE']:.2f} for {LBL[lose]} and {base_m['MAE']:.2f} for the 7-day baseline — {mae_gain:.0f}% lower than the "
            f"baseline in MAE and {rmse_gain:.0f}% lower in RMSE." + close_note
        )
    else:
        st.error(f"Neither model beats the baseline for {sku_choice} on the 30% test data — this needs fixing.")

    st.info(
        "**Why the baseline's calm tracking signal doesn't make it better:** the signal divides by average error, and the "
        "baseline's errors are roughly twice as large, so its signal looks tame while the forecast itself is far less accurate. "
        "That is why we judge on **accuracy and tracking signal together**."
    )

    # ------------------------------------------------------------ 70 vs 30
    st.subheader("70% training data vs 30% test data — side by side")
    sbs = pd.DataFrame([
        {"Metric": m,
         "Random Forest (tuned) — 70% training": f"{bundle['rf_metrics_train'][m]:.2f}",
         "Random Forest (tuned) — 30% test": f"{bundle['rf_tuned_metrics'][m]:.2f}",
         "Gradient Boosting (tuned) — 70% training": f"{bundle['gb_metrics_train'][m]:.2f}",
         "Gradient Boosting (tuned) — 30% test": f"{bundle['gb_tuned_metrics'][m]:.2f}",
         "Baseline — 70% training": f"{bundle['baseline_metrics_train'][m]:.2f}",
         "Baseline — 30% test": f"{bundle['baseline_metrics'][m]:.2f}"}
        for m in ("MAE", "RMSE", "MAPE")
    ])
    st.dataframe(sbs, use_container_width=True, hide_index=True)
    st.caption(
        "The 70% columns are data the models learned from, so they look better there. The **30% columns are the real test**; "
        "a small gap between the two means the model is not overfitting. "
        "(The online bias correction is not included in this table — it only exists while forecasting day by day.)"
    )

    # ------------------------------------------------------------ 2. tracking signal, both models
    st.subheader("2️⃣ Tracking Signal (RSFE ÷ cumulative MAD) — both models over the 30% test file")
    st.markdown(
        "The tracking signal shows whether a forecast is *going around the actual* (staying near zero) or drifting "
        "consistently one direction — which means the model is **over- or under-estimating**. A single MAE number can hide "
        "that. Conventional control limit: **±4**."
    )
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["rf_tracking_signal"],
                                name="Random Forest — FINAL", line=dict(color="green", width=2.5)))
    fig_ts.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["gb_tracking_signal"],
                                name="Gradient Boosting — FINAL", line=dict(color="royalblue", width=2.5)))
    fig_ts.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["baseline_tracking_signal"],
                                name="Baseline (7-day MA)", line=dict(color="orange", width=1.5)))
    fig_ts.add_hline(y=ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_ts.add_hline(y=-ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_ts.add_hline(y=0, line_color="grey", line_width=1)
    fig_ts.update_layout(title="Tracking Signal over the 30% test file (2025-05 → 2026-04)", yaxis_title="RSFE / MAD")
    st.plotly_chart(fig_ts, use_container_width=True)

    tsc1, tsc2, tsc3 = st.columns(3)
    tsc1.metric("Random Forest FINAL — days beyond ±4", bundle["rf_ts_breaches"],
                delta=f"{bundle['rf_ts_breaches'] - bundle['rf_original_ts_breaches']:+d} vs Sprint 3 original",
                delta_color="inverse")
    tsc2.metric("Gradient Boosting FINAL — days beyond ±4", bundle["gb_ts_breaches"],
                delta=f"{bundle['gb_ts_breaches'] - bundle['gb_tuned_ts_breaches']:+d} vs no correction",
                delta_color="inverse")
    tsc3.metric("Baseline — days beyond ±4", bundle["baseline_ts_breaches"])
    st.caption(
        f"Fair-yardstick note: because the signal accumulates, even a **perfectly unbiased** forecaster with this much "
        f"day-to-day noise would still sit outside ±4 on about **{bundle['unbiased_reference_breach_pct']:.0f}%** of days over a "
        f"full year (it wanders like a random walk). So the question is not 'zero breaches' but 'does it drift steadily one way?' "
        f"Before the correction both models did (Random Forest original: {bundle['rf_original_ts_breaches']} days, "
        f"Gradient Boosting without correction: {bundle['gb_tuned_ts_breaches']} days); with it they stay around zero."
    )

    # ------------------------------------------------------------ 3. is the winner reliable by itself?
    st.subheader(f"3️⃣ Is the winning model ({LBL[win]}) reliable by itself? — bias check")
    st.markdown(
        "Beating another model only says which is *better*. To show the model can be **trusted**, we check its errors for "
        "bias: over the 30% test file and over the **historical** window (the last "
        f"{len(bundle['history']['actual'])} days of the 70%, forecast by models fitted only on the days before them). "
        "If the errors are random around zero with no trend, the forecast neither over- nor under-estimates."
    )

    def _verdict(b, breach_share):
        leaning = abs(b["t_stat"]) >= 2
        streaky = abs(b["autocorr_lag1"]) >= 0.2
        if leaning and b["t_stat"] > 0:
            return "Leans LOW (under-forecasts)"
        if leaning:
            return "Leans HIGH (over-forecasts)"
        if breach_share > 0.5:
            return f"Tends to drift (signal outside ±4 on {breach_share:.0%} of days)"
        if streaky:
            return "Centred, but errors come in streaks"
        return "✅ No systematic bias"

    def _breach_share(actual, pred):
        ts = ml_forecast._tracking_signal(actual, pred)
        return float(np.mean(np.abs(ts) > ml_forecast.TRACKING_SIGNAL_LIMIT))

    _ta = bundle["test_actual"]
    test_rows = [
        (LBL["rf"] + " — FINAL", ml_forecast._bias_summary(_ta, bundle["rf_pred"]), _breach_share(_ta, bundle["rf_pred"])),
        (LBL["gb"] + " — FINAL", ml_forecast._bias_summary(_ta, bundle["gb_pred"]), _breach_share(_ta, bundle["gb_pred"])),
        (LBL["rf"] + " — without correction", ml_forecast._bias_summary(_ta, bundle["rf_tuned_pred"]),
         _breach_share(_ta, bundle["rf_tuned_pred"])),
        (LBL["gb"] + " — without correction", ml_forecast._bias_summary(_ta, bundle["gb_tuned_pred"]),
         _breach_share(_ta, bundle["gb_tuned_pred"])),
        ("7-day moving average (baseline)", bundle["bias_test"]["baseline"], _breach_share(_ta, bundle["baseline_pred"])),
    ]
    hist = bundle["history"]
    bias_rows = [{
        "Forecast": name, "Window": "30% test", "Days": b["n"],
        "Avg error / day": f"{b['mean_error']:+.2f}", "t-statistic": f"{b['t_stat']:+.1f}",
        "Day-to-day error correlation": f"{b['autocorr_lag1']:+.2f}",
        "Days outside ±4": f"{share:.0%}", "Verdict": _verdict(b, share),
    } for name, b, share in test_rows]
    bias_rows += [{
        "Forecast": LBL[k] + " — FINAL", "Window": "Historical (end of 70%)", "Days": hist[k + "_bias"]["n"],
        "Avg error / day": f"{hist[k + '_bias']['mean_error']:+.2f}", "t-statistic": f"{hist[k + '_bias']['t_stat']:+.1f}",
        "Day-to-day error correlation": f"{hist[k + '_bias']['autocorr_lag1']:+.2f}",
        "Days outside ±4": f"{float(np.mean(np.abs(hist[k + '_ts']) > ml_forecast.TRACKING_SIGNAL_LIMIT)):.0%}",
        "Verdict": _verdict(hist[k + "_bias"], float(np.mean(np.abs(hist[k + "_ts"]) > ml_forecast.TRACKING_SIGNAL_LIMIT))),
    } for k in ("rf", "gb")]
    st.dataframe(pd.DataFrame(bias_rows), use_container_width=True, hide_index=True)
    st.caption(
        "t-statistic = average error ÷ its standard error; |t| below 2 means the average error is statistically "
        "indistinguishable from zero (a conservative test, so a small steady lean can still pass it). Day-to-day error "
        "correlation near 0 means the errors look random; a large positive value means streaks (a trend the model is "
        "missing). 'Days outside ±4' is the share of days the tracking signal leaves its limits — it is more sensitive to a "
        "persistent lean, which is why the 'without correction' rows are flagged as drifting. The adaptive correction pulls the "
        "average error to about zero by design, so read the verdict together with the tracking signal and rolling error."
    )

    err_dates = pd.to_datetime(bundle["test_dates"])
    roll = lambda pred: pd.Series(bundle["test_actual"] - pred, index=err_dates).rolling(30).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=err_dates, y=roll(bundle["rf_pred"]), name="Random Forest — FINAL",
                                  line=dict(color="green", width=2.5)))
    fig_roll.add_trace(go.Scatter(x=err_dates, y=roll(bundle["gb_pred"]), name="Gradient Boosting — FINAL",
                                  line=dict(color="royalblue", width=2.5)))
    fig_roll.add_trace(go.Scatter(x=err_dates, y=roll(bundle[win + "_tuned_pred"]),
                                  name=f"{LBL[win]} — without correction",
                                  line=dict(color="crimson", width=1.5, dash="dot")))
    fig_roll.add_hline(y=0, line_color="grey", line_width=1)
    fig_roll.update_layout(
        title="Rolling 30-day average error (actual − forecast): around zero = no over- or under-estimating",
        yaxis_title="units / day")
    st.plotly_chart(fig_roll, use_container_width=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist["dates"], y=hist["rf_ts"], name="Random Forest — FINAL",
                                  line=dict(color="green", width=2.5)))
    fig_hist.add_trace(go.Scatter(x=hist["dates"], y=hist["gb_ts"], name="Gradient Boosting — FINAL",
                                  line=dict(color="royalblue", width=2.5)))
    fig_hist.add_hline(y=ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_hist.add_hline(y=-ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_hist.add_hline(y=0, line_color="grey", line_width=1)
    fig_hist.update_layout(
        title="Tracking signal on the historical window (end of the 70%, forecast out-of-sample)", yaxis_title="RSFE / MAD")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🔴 Tracking signal keeps updating in Live Simulation (after the 30% window)")
    st.caption(
        "Each simulated day is forecast one step ahead by the models trained on the 70% file (with the same adaptive bias "
        "correction) and compared with what actually sold. The signal keeps accumulating for as long as the simulation "
        f"runs — including past the end of the 30% test window ({TEST_DF['Date'].max().date()})."
    )
    if LIVE_MODE:
        live_ts = ml_forecast.live_tracking_signal(df, sku_choice, bundle, HISTORICAL_AS_OF_DATE + pd.Timedelta(days=1))
        if len(live_ts["dates"]) == 0:
            st.info("Run a few more simulated days to start the live tracking signal.")
        else:
            fig_live_ts = go.Figure()
            fig_live_ts.add_trace(go.Scatter(x=live_ts["dates"], y=live_ts["rf"], name="Random Forest — FINAL", line=dict(color="green")))
            fig_live_ts.add_trace(go.Scatter(x=live_ts["dates"], y=live_ts["gb"], name="Gradient Boosting — FINAL", line=dict(color="royalblue")))
            fig_live_ts.add_trace(go.Scatter(x=live_ts["dates"], y=live_ts["baseline"], name="Baseline (7-day MA)", line=dict(color="orange")))
            fig_live_ts.add_hline(y=ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
            fig_live_ts.add_hline(y=-ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
            end30 = str(TEST_DF["Date"].max().date())
            if pd.Timestamp(live_ts["dates"][-1]) > TEST_DF["Date"].max():
                fig_live_ts.add_shape(type="line", x0=end30, x1=end30, y0=0, y1=1, yref="paper",
                                       line=dict(color="grey", dash="dash"))
                fig_live_ts.add_annotation(x=end30, y=1, yref="paper", text="30% test window ends",
                                            showarrow=False, yshift=10)
            fig_live_ts.update_layout(title=f"Live tracking signal — {len(live_ts['dates'])} simulated days so far",
                                       yaxis_title="RSFE / MAD")
            st.plotly_chart(fig_live_ts, use_container_width=True)
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Simulated days tracked", len(live_ts["dates"]))
            lc2.metric("Random Forest — latest signal", f"{live_ts['rf'][-1]:.1f}")
            lc3.metric("Gradient Boosting — latest signal", f"{live_ts['gb'][-1]:.1f}")
            lc4.metric("Baseline — latest signal", f"{live_ts['baseline'][-1]:.1f}")
    else:
        st.info("Go to 🔴 Live Simulation and run some days — the live tracking signal will appear here and keep updating.")

    with st.expander("How the forecasts were built (and what the correction costs)"):
        st.markdown(
            "Every choice below was made using **only the 70% training file** — the 30% test file was never used to pick settings.\n\n"
            "1. **Random Forest.** Regularised (`min_samples_leaf=15`, `max_features=0.6`, picked on a validation slice at the end of "
            "the 70%) so the trees stop memorising day-to-day noise, then **refit on all of the 70%**.\n"
            f"2. **Gradient Boosting.** Trees are built one after another, each correcting the previous ones' errors. Settings picked "
            f"on the same validation slice (lowest validation MAE {bundle['gb_val_mae']:.2f}): `n_estimators={bundle['gb_params']['n_estimators']}`, "
            f"`learning_rate={bundle['gb_params']['learning_rate']}`, `max_depth={bundle['gb_params']['max_depth']}`, "
            f"`min_samples_leaf={bundle['gb_params']['min_samples_leaf']}`; then refit on all of the 70%.\n"
            "3. **Adaptive bias correction (both models).** When the tracking signal shows a model drifting one way, each next "
            "forecast is nudged back by a smoothed average of its recent errors (updated only *after* each day ends — a day is "
            f"never used to forecast itself). Smoothing rate: Random Forest **{bundle['alpha']:.2f}**, Gradient Boosting "
            f"**{bundle['gb_alpha']:.2f}**, the gentlest rate that kept the validation tracking signal inside ±4 on at least 90% of days.\n"
        )
        st.markdown(
            f"**Cost of the correction:** Random Forest MAE {bundle['rf_tuned_metrics']['MAE']:.2f} → {bundle['rf_metrics']['MAE']:.2f}; "
            f"Gradient Boosting MAE {bundle['gb_tuned_metrics']['MAE']:.2f} → {bundle['gb_metrics']['MAE']:.2f} — a small accuracy "
            f"trade for forecasts that no longer drift. Both are still well ahead of the baseline ({base_m['MAE']:.2f})."
        )

    with st.expander("Feature importances (what each model actually learned from)"):
        fi_a, fi_b = st.columns(2)
        fi_a.caption("Random Forest")
        fi_a.bar_chart(pd.Series(bundle["feature_importances"]).sort_values(ascending=False))
        fi_b.caption("Gradient Boosting")
        fi_b.bar_chart(pd.Series(bundle["gb_feature_importances"]).sort_values(ascending=False))

    st.subheader(f"Forward forecast — next {forecast_period} days (recursive, network-wide)")
    fwd_rf = ml_forecast.forecast_forward(df, sku_choice, bundle, AS_OF_DATE, forecast_period, which="rf")
    fwd_gb = ml_forecast.forecast_forward(df, sku_choice, bundle, AS_OF_DATE, forecast_period, which="gb")
    fwd = fwd_rf.rename(columns={"ML_Forecast": "Random Forest"}).merge(
        fwd_gb.rename(columns={"ML_Forecast": "Gradient Boosting"}), on="Date")
    fwd["Date"] = fwd["Date"].dt.strftime("%Y-%m-%d")
    fwd[["Random Forest", "Gradient Boosting"]] = fwd[["Random Forest", "Gradient Boosting"]].round(1)
    st.dataframe(fwd, use_container_width=True, hide_index=True)
    st.caption(f"Winner on the 30% test file: **{LBL[win]}**. Recursive forecast: each predicted day feeds the lag features "
               "for the next day. Multi-step forecasts like this naturally get less certain the further out they go.")

# ============================================================== 4. INVENTORY MONITORING
with tab_inventory:
    st.header("Inventory Monitoring — All Stores × All SKUs")
    st.caption(f"As of {AS_OF_DATE.date()}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Opening Stock", f"{full_table['Opening_Stock'].sum():,.0f}")
    c2.metric("Total Incoming Today", f"{full_table['Incoming_Stock'].sum():,.0f}")
    c3.metric("Total Units Sold Today", f"{full_table['Units_Sold'].sum():,.0f}")
    c4.metric("Total Closing Stock", f"{full_table['Closing_Stock'].sum():,.0f}")

    display_cols = ["Store", "Product", "SKU", "Opening_Stock", "Incoming_Stock", "Units_Sold",
                     "Closing_Stock", "Wastage", "Stockout", "Risk"]
    st.dataframe(full_table[display_cols], use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(full_table, x="Store", y="Closing_Stock", color="Product",
                     barmode="group", title="Closing Stock by Store, split by SKU")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(full_table, x="Store", y="Wastage", color="Product",
                     barmode="group", title="Today's Wastage by Store, split by SKU")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== STOCK INTAKE
with tab_intake:
    st.header("📥 Stock Intake — Order & Log Received Stock")

    if "placed_orders" not in st.session_state:
        st.session_state.placed_orders = []

    # ---- Recommended Orders (Order Management), always at the top ----
    st.subheader("📋 Recommended Orders right now")
    if not LIVE_MODE:
        st.caption(
            "Calculated live from the current forecast, stock and safety-stock rate. Placing an order "
            "here is recorded with its expected lead time, but since you're viewing the static "
            "historical data (Live Simulation hasn't been started), the order won't actually arrive "
            "and change stock — start Live Simulation to see placed orders arrive and affect every "
            "other tab."
        )
    else:
        st.caption(
            "Calculated live from the current forecast, stock, real incoming pipeline and each "
            "combo's adaptively learned safety-stock rate. Placing an order here queues it into the "
            "same pipeline the Live Simulation uses — it will actually arrive after its lead time."
        )

    needs_order = full_table[full_table["Recommended_Order"] > 0].sort_values("Recommended_Order", ascending=False)
    if needs_order.empty:
        st.success("✅ No store × SKU combination currently needs an order.")
    else:
        for _, r in needs_order.iterrows():
            rc1, rc2, rc3, rc4, rc5 = st.columns([2, 2, 1.3, 1.3, 1.4])
            rc1.markdown(f"**{r['Store']}**")
            rc2.markdown(f"{r['Product']} ({r['SKU']})")
            rc3.markdown(f"Order **{r['Recommended_Order']:.0f}** units")
            rc4.markdown(f"Lead time: **{r['Lead_Time_Days']:.0f}d**")
            order_key = f"place_order_{r['Store']}_{r['SKU']}"
            if rc5.button("Place order", key=order_key):
                qty = int(r["Recommended_Order"])
                lead_time = int(r["Lead_Time_Days"])
                if LIVE_MODE:
                    result = simulation.place_manual_order(sim_state, r["Store"], r["SKU"], qty, lead_time)
                    arrival = result["arrival_date"]
                else:
                    arrival = AS_OF_DATE + pd.Timedelta(days=lead_time)
                st.session_state.placed_orders.append({
                    "store": r["Store"], "sku": r["SKU"], "qty": qty,
                    "date_placed": str(AS_OF_DATE.date()), "lead_time_days": lead_time,
                    "expected_arrival": str(pd.Timestamp(arrival).date()), "live": LIVE_MODE,
                })
                st.success(f"Ordered {qty} units of {r['SKU']} for {r['Store']} — "
                           f"lead time {lead_time} day(s), expected {pd.Timestamp(arrival).date()}.")
                st.rerun()

    if st.session_state.placed_orders:
        st.subheader("Placed orders (this session)")
        po_df = pd.DataFrame(st.session_state.placed_orders)
        st.dataframe(po_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Log stock as it's physically received")
    st.markdown(
        "Use this once stock actually arrives at the store (whether from a placed order above or "
        "any other delivery). Every entry adds directly to that Store × SKU's Current Stock and "
        "immediately flows through to Lead-Time, Reorder, Risk and every other tab in this app. "
        "**The quantity below updates its own overstock check live, before you submit anything.**"
    )

    intake_store = st.selectbox("Store", STORES, key="intake_store")
    product_labels = [f"{sku} – {PRODUCT_NAME[sku]}" for sku in PRODUCTS]
    intake_product_label = st.selectbox("Product", product_labels, key="intake_product")
    intake_sku = intake_product_label.split(" – ")[0]
    if "intake_qty" not in st.session_state:
        st.session_state["intake_qty"] = 50
    intake_qty = st.number_input("Quantity received", min_value=0, step=1, key="intake_qty")
    intake_note = st.text_input("Note (optional)", key="intake_note")

    row = full_table[(full_table["Store"] == intake_store) & (full_table["SKU"] == intake_sku)]
    if not row.empty:
        r = row.iloc[0]
        hypothetical_stock = r["Current_Stock"] + intake_qty
        assessment = inv.assess_stock_position(hypothetical_stock, r["Forecast_During_Lead_Time"], r["Safety_Stock"])
        if assessment["status"] == "Overstocked":
            st.warning(f"⚠️ **This would overstock it.** {assessment['message']}")
        elif assessment["status"] == "Understocked":
            st.info(f"ℹ️ {assessment['message']}")
        else:
            st.success(f"✅ **Looks right.** {assessment['message']}")

    if st.button("➕ Log stock receipt", type="primary"):
        st.session_state.stock_ledger.append({
            "store": intake_store, "sku": intake_sku, "qty": int(intake_qty), "note": intake_note,
        })
        st.session_state["_intake_pending_reset"] = {"intake_qty": 50, "intake_note": ""}
        st.success(f"Logged {intake_qty} units of {intake_sku} at {intake_store}.")
        st.rerun()

    st.divider()
    st.subheader("🤖 AI Stock Assessment")
    st.caption("Reads the same demand forecast and reorder-point numbers already calculated "
               "elsewhere in the app — no separate model, no new assumptions.")
    a_store = st.selectbox("Store ", STORES, key="assess_store")
    a_product_label = st.selectbox("Product ", product_labels, key="assess_product")
    a_sku = a_product_label.split(" – ")[0]
    a_row = full_table[(full_table["Store"] == a_store) & (full_table["SKU"] == a_sku)]
    if not a_row.empty:
        r = a_row.iloc[0]
        assessment = inv.assess_stock_position(r["Current_Stock"], r["Forecast_During_Lead_Time"], r["Safety_Stock"])
        st.markdown(f"### {assessment['status']}")
        st.write(assessment["message"])
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Current Stock", f"{r['Current_Stock']:.0f}")
        cB.metric("Forecast During Lead Time", f"{r['Forecast_During_Lead_Time']:.0f}")
        cC.metric("Reorder Point", f"{assessment['reorder_point']:.0f}")
        cD.metric("Overstock Threshold", f"{assessment['overstock_threshold']:.0f}")

    st.divider()
    st.subheader("Intake ledger (this session)")
    if st.session_state.stock_ledger:
        st.dataframe(pd.DataFrame(st.session_state.stock_ledger), use_container_width=True, hide_index=True)
    else:
        st.caption("No manual stock entries logged yet.")

# ============================================================== MANUAL ORDER (AI-CHECKED)
with tab_order:
    st.header("📦 Manual Order — the AI checks your order before you place it")
    st.markdown(
        "Enter how many units you want to order. The AI compares it with the **demand forecast**, the "
        "**anticipated supplier lead time**, the **reorder point** (forecast during lead time + safety stock) "
        "and what is **already on order**, then tells you whether the order is **right, overstocking or "
        "understocking** — before anything is placed."
    )

    o_store = st.selectbox("Store", STORES, key="order_store")
    o_labels = [f"{sku} – {PRODUCT_NAME[sku]}" for sku in PRODUCTS]
    o_label = st.selectbox("Product", o_labels, key="order_product")
    o_sku = o_label.split(" – ")[0]

    o_row = full_table[(full_table["Store"] == o_store) & (full_table["SKU"] == o_sku)]
    if o_row.empty:
        st.warning("No data for that Store × SKU.")
    else:
        r = o_row.iloc[0]
        planned_lt = int(r["Lead_Time_Days"])
        ant = simulation.anticipate_lead_time(sim_state["lead_time_log"], o_store, o_sku, planned_lt)
        lt_used = ant["anticipated_days"]

        # Stock already on order: the real pipeline in Live Mode; in static mode, the estimate
        # plus any orders placed this session (which are recorded but do not arrive in static view).
        incoming_qty = float(r["Incoming_Stock_Estimate"])
        if not LIVE_MODE:
            incoming_qty += sum(o["qty"] for o in st.session_state.placed_orders
                                if o["store"] == o_store and o["sku"] == o_sku and not o["live"])

        base_check = inv.assess_order(r["Current_Stock"], incoming_qty, 0, r["Avg_Daily_Forecast"], lt_used,
                                      r["Safety_Stock_Rate"], r["Shelf_Life_Days"])
        order_qty = st.number_input(
            "Order quantity (units)", min_value=0, step=1,
            value=int(base_check["recommended_order"]), key=f"order_qty_{o_store}_{o_sku}",
            help="Pre-filled with the AI's recommended quantity — change it to test your own number.",
        )
        chk = inv.assess_order(r["Current_Stock"], incoming_qty, order_qty, r["Avg_Daily_Forecast"], lt_used,
                               r["Safety_Stock_Rate"], r["Shelf_Life_Days"])

        # ---- the verdict
        banner = f"### 🤖 AI verdict: {chk['verdict']}"
        if chk["level"] == "success":
            st.success(banner)
        elif chk["level"] == "warning":
            st.warning(banner)
        elif chk["level"] == "error":
            st.error(banner)
        else:
            st.info(banner)
        st.write(chk["explanation"])
        if chk["expiry_warning"]:
            st.warning("⏳ " + chk["expiry_warning"])

        diff = chk["order_vs_recommended"]
        if chk["recommended_order"] == 0 and order_qty == 0:
            st.caption("AI recommended order: 0 units.")
        elif diff == 0:
            st.caption(f"Your order matches the AI's recommended order ({chk['recommended_order']} units).")
        elif diff > 0:
            st.caption(f"AI recommended order: **{chk['recommended_order']} units** — yours is {diff} unit(s) more.")
        else:
            st.caption(f"AI recommended order: **{chk['recommended_order']} units** — yours is {abs(diff)} unit(s) fewer.")

        # ---- the numbers behind it
        st.subheader("Numbers the AI used")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current stock", f"{r['Current_Stock']:.0f}")
        m2.metric("Already on order", f"{incoming_qty:.0f}")
        m3.metric("Avg daily demand forecast", f"{r['Avg_Daily_Forecast']:.1f}")
        m4.metric("Supplier lead time used", f"{lt_used} day(s)")
        st.caption(f"Lead time basis: {ant['basis']}. Planned lead time is {planned_lt} day(s).")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Forecast during lead time", f"{chk['forecast_during_lead_time']:.0f}")
        n2.metric("Reorder point", f"{chk['reorder_point']:.0f}",
                  help="Forecast during lead time + safety stock")
        n3.metric("Overstock threshold", f"{chk['overstock_threshold']:.0f}",
                  help="2× forecast demand during lead time")
        n4.metric("Stock position after order", f"{chk['position_after']:.0f}",
                  delta=(f"{chk['days_cover']:.1f} days of sales" if chk["days_cover"] is not None else None),
                  delta_color="off")

        expected_arrival = AS_OF_DATE + pd.Timedelta(days=lt_used)
        st.caption(f"If placed today ({AS_OF_DATE.date()}), the AI expects delivery around "
                   f"**{expected_arrival.date()}**.")

        if not LIVE_MODE:
            st.caption("Static view: placed orders are recorded here and counted as 'already on order' for the "
                       "next check, but they only actually arrive if you start the 🔴 Live Simulation.")

        if st.button("✅ Place this order", type="primary", disabled=(order_qty <= 0), key="place_manual_order_btn"):
            qty = int(order_qty)
            if LIVE_MODE:
                res = simulation.place_manual_order(sim_state, o_store, o_sku, qty, planned_lt)
                arrival = res["arrival_date"]
            else:
                arrival = AS_OF_DATE + pd.Timedelta(days=planned_lt)
            st.session_state.placed_orders.append({
                "store": o_store, "sku": o_sku, "qty": qty,
                "date_placed": str(AS_OF_DATE.date()), "lead_time_days": planned_lt,
                "expected_arrival": str(pd.Timestamp(arrival).date()), "live": LIVE_MODE,
                "ai_verdict": chk["verdict"],
            })
            st.success(f"Ordered {qty} units of {o_sku} for {o_store} (AI verdict: {chk['verdict']}). "
                       f"Lead time {planned_lt} day(s), expected {pd.Timestamp(arrival).date()}.")
            st.rerun()

    if st.session_state.placed_orders:
        st.subheader("Orders placed this session")
        st.dataframe(pd.DataFrame(st.session_state.placed_orders), use_container_width=True, hide_index=True)

# ============================================================== 5. LEAD-TIME TRACKING
with tab_leadtime:
    st.header("Lead-Time Tracking — All Stores × All SKUs")
    st.caption("Incoming stock shown here is a SIMULATED ESTIMATE, since the dataset does not record pending purchase orders explicitly.")
    cols = ["Store", "Product", "SKU", "Lead_Time_Days", "Forecast_During_Lead_Time", "Current_Stock", "Incoming_Stock_Estimate"]
    st.dataframe(
        full_table[cols].rename(columns={
            "Lead_Time_Days": "Supplier Lead Time (days)",
            "Forecast_During_Lead_Time": "Forecasted Demand During Lead Time",
            "Incoming_Stock_Estimate": "Incoming Stock (est.)",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"As of {AS_OF_DATE.date()} · Forecast period = {forecast_period} days.")

    fig = px.bar(full_table, x="Store", y=["Current_Stock", "Incoming_Stock_Estimate", "Forecast_During_Lead_Time"],
                 facet_col="Product", barmode="group",
                 title="Current Stock vs Forecasted Demand During Lead Time, by Store & SKU")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================== LEAD-TIME ANTICIPATION
with tab_leadtime_anticipation:
    st.header("🚚 Lead-Time Anticipation — supplier delivery forecast")
    st.markdown(
        "The dataset records one **fixed, planned** lead time per SKU. Real suppliers are sometimes late. "
        "This tab **anticipates when each supplier delivery will really arrive**, using the deliveries actually "
        "observed in the Live Simulation, and shows how much **extra demand** a slower supplier forces you to cover. "
        "The anticipated lead time also feeds the AI check in the 📦 **Manual Order** tab."
    )
    st.caption("Suppliers are not named in the data, so each Store × SKU supply line is treated as one supplier. "
               f"With fewer than {simulation.MIN_LEAD_TIME_OBSERVATIONS} observed deliveries the planned lead time is used.")

    lt_log = sim_state["lead_time_log"]
    rows = []
    for _, r in full_table.iterrows():
        planned = int(r["Lead_Time_Days"])
        a = simulation.anticipate_lead_time(lt_log, r["Store"], r["SKU"], planned)
        ant_days = a["anticipated_days"]
        p80_days = int(np.ceil(a["p80_actual"]))
        daily = r["Avg_Daily_Forecast"]
        rows.append({
            "Store": r["Store"], "Product": r["Product"], "SKU": r["SKU"],
            "Planned lead time (d)": planned,
            "Anticipated lead time (d)": ant_days,
            "Worst likely (80th pct, d)": p80_days,
            "On-time deliveries": ("–" if a["on_time_pct"] is None else f"{a['on_time_pct']:.0%}"),
            "Deliveries observed": a["n_observations"],
            "Expected arrival if ordered today": str((AS_OF_DATE + pd.Timedelta(days=ant_days)).date()),
            "Latest likely arrival": str((AS_OF_DATE + pd.Timedelta(days=max(p80_days, ant_days))).date()),
            "Demand over planned lead time": round(daily * planned, 1),
            "Demand over anticipated lead time": round(daily * ant_days, 1),
            "Extra units to cover if supplier is late": round(daily * (ant_days - planned), 1),
        })
    ant_df = pd.DataFrame(rows)

    if lt_log.empty:
        st.info("🚚 No delivery history yet — open the 🔴 **Live Simulation** tab and advance some days (or turn on "
                "Auto-play). As orders are placed and arrive, this tab learns each supplier's real delivery "
                "pattern. Until then the anticipated lead time simply equals the planned one.")
    else:
        all_actual = lt_log["Actual_Lead_Time"].astype(float)
        merged = lt_log.merge(full_table[["Store", "SKU", "Lead_Time_Days"]],
                              left_on=["Store", "Product_ID"], right_on=["Store", "SKU"], how="left")
        on_time = float((merged["Actual_Lead_Time"] <= merged["Lead_Time_Days"]).mean())
        delay = float((merged["Actual_Lead_Time"] - merged["Lead_Time_Days"]).clip(lower=0).mean())
        k1, k2, k3 = st.columns(3)
        k1.metric("Deliveries tracked", f"{len(lt_log)}")
        k2.metric("Delivered on time", f"{on_time:.0%}")
        k3.metric("Average delay vs plan", f"{delay:.2f} day(s)")

    st.subheader("Anticipated delivery by Store × SKU")
    st.dataframe(ant_df, use_container_width=True, hide_index=True)

    if not lt_log.empty:
        plot_df = lt_log.copy()
        plot_df["Product"] = plot_df["Product_ID"].map(PRODUCT_NAME)
        fig = px.histogram(plot_df, x="Actual_Lead_Time", color="Product", barmode="group",
                           title="Actual supplier lead times observed (days)",
                           labels={"Actual_Lead_Time": "Actual lead time (days)"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars to the right of a product's planned lead time are late deliveries. "
                   "Extra demand to cover = average daily forecast × (anticipated − planned) days.")

# ============================================================== 6. EXPIRY TRACKING
with tab_expiry:
    st.header("Expiry Tracking — All Stores × All SKUs")
    st.warning("⚠️ **Simulated/prototype values.** This dataset is not connected to Colonial Fresh's "
               "live operational systems. Days Remaining reflects the shelf life of the most recently "
               "received stock lot as of the dataset's latest date — a simplification appropriate for this design.")
    cols = ["Store", "Product", "SKU", "Shelf_Life_Days", "Expiry_Date", "Days_Remaining", "Risk"]
    expiry_view = full_table[cols].copy()
    expiry_view["Expiry Risk"] = np.where(expiry_view["Days_Remaining"] <= inv.EXPIRY_RISK_DAYS_THRESHOLD, "Watch", "OK")
    st.dataframe(expiry_view.drop(columns=["Risk"]), use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()}.")

# ============================================================== 7. REORDER RECOMMENDATION
with tab_reorder:
    st.header("🧾 Order Management Dashboard")
    st.caption(f"Every store × SKU's order status as of {AS_OF_DATE.date()}, calculated live from the demand forecast, "
               "anticipated supplier lead time, safety stock and inventory position — not manually maintained.")
    st.markdown(
        "**Recommended Order = Forecast During Lead Time + Safety Stock − Current Stock − Incoming Stock** "
        "(floored at 0). Safety Stock = adaptively learned rate × Forecast During Lead Time "
        "(20% baseline until Live Simulation has learned otherwise)."
    )

    st.caption("Quantities use the **anticipated** supplier lead time (see 🚚 Lead-Time Anticipation). Where that is longer "
               "than the planned lead time, the figure can be higher than the one shown in 📥 Stock Intake.")

    # ---- build the order-management view of the network
    om_rows = []
    for _, r in full_table.iterrows():
        planned = int(r["Lead_Time_Days"])
        ant = simulation.anticipate_lead_time(sim_state["lead_time_log"], r["Store"], r["SKU"], planned)
        lt_used = ant["anticipated_days"]
        chk0 = inv.assess_order(r["Current_Stock"], r["Incoming_Stock_Estimate"], 0, r["Avg_Daily_Forecast"],
                                lt_used, r["Safety_Stock_Rate"], r["Shelf_Life_Days"])
        need = chk0["recommended_order"]
        chk = inv.assess_order(r["Current_Stock"], r["Incoming_Stock_Estimate"], need, r["Avg_Daily_Forecast"],
                               lt_used, r["Safety_Stock_Rate"], r["Shelf_Life_Days"])
        if need > 0:
            urgent = r["Current_Stock"] + r["Incoming_Stock_Estimate"] < chk["forecast_during_lead_time"]
            priority = "🔴 Urgent" if urgent else "🟡 Routine"
        else:
            priority = "🟢 No order needed"
        om_rows.append({
            "Store": r["Store"], "Product": r["Product"], "SKU": r["SKU"],
            "Priority": priority, "Risk": r["Risk"],
            "Current Stock": round(r["Current_Stock"]), "On Order": round(r["Incoming_Stock_Estimate"]),
            "Forecast During Lead Time": round(chk["forecast_during_lead_time"], 1),
            "Reorder Point": round(chk["reorder_point"], 1),
            "Recommended Order": need,
            "Lead Time (d)": lt_used,
            "Expected Arrival": str((AS_OF_DATE + pd.Timedelta(days=lt_used)).date()) if need > 0 else "–",
            "AI Check": chk["verdict"],
            "_planned_lt": planned,
        })
    om = pd.DataFrame(om_rows)
    to_order = om[om["Recommended Order"] > 0].copy()
    to_order["_sort"] = to_order["Priority"].map({"🔴 Urgent": 0, "🟡 Routine": 1})
    to_order = to_order.sort_values(["_sort", "Recommended Order"], ascending=[True, False])

    if LIVE_MODE:
        on_order_units = sum(sum(c["pending_orders"].values()) for c in sim_state["combos"].values())
    else:
        on_order_units = float(full_table["Incoming_Stock_Estimate"].sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Orders needed now", f"{len(to_order)} / {len(om)}")
    k2.metric("Units to order", f"{int(to_order['Recommended Order'].sum()):,}")
    k3.metric("🔴 Urgent lines", int((to_order["Priority"] == "🔴 Urgent").sum()))
    k4.metric("Stockout-risk lines", int((om["Risk"] == "Stockout Risk").sum()))
    k5.metric("Overstock-risk lines", int((om["Risk"] == "Overstock Risk").sum()))
    k6.metric("Units already on order", f"{on_order_units:,.0f}")

    # ---- 1. orders to place
    st.subheader("1 · Orders to place now")
    if to_order.empty:
        st.success("✅ No store × SKU combination needs an order right now.")
    else:
        show_cols = ["Priority", "Store", "Product", "Current Stock", "On Order", "Forecast During Lead Time",
                     "Reorder Point", "Recommended Order", "Lead Time (d)", "Expected Arrival", "AI Check"]
        st.dataframe(to_order[show_cols], use_container_width=True, hide_index=True)
        if st.button("📨 Place ALL recommended orders", key="place_all_btn"):
            for _, o in to_order.iterrows():
                qty = int(o["Recommended Order"])
                if LIVE_MODE:
                    res = simulation.place_manual_order(sim_state, o["Store"], o["SKU"], qty, int(o["_planned_lt"]))
                    arrival = res["arrival_date"]
                else:
                    arrival = AS_OF_DATE + pd.Timedelta(days=int(o["_planned_lt"]))
                st.session_state.placed_orders.append({
                    "store": o["Store"], "sku": o["SKU"], "qty": qty, "date_placed": str(AS_OF_DATE.date()),
                    "lead_time_days": int(o["_planned_lt"]), "expected_arrival": str(pd.Timestamp(arrival).date()),
                    "live": LIVE_MODE, "ai_verdict": o["AI Check"],
                })
            st.success(f"Placed {len(to_order)} order(s).")
            st.rerun()
        st.caption("Want to change a quantity first? Use the 📦 **Manual Order** tab — the AI will tell you if your number "
                   "is right, overstocking or understocking.")

    # ---- 2. charts
    st.subheader("2 · Where the orders are needed")
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.bar(om, x="Store", y="Recommended Order", color="Product", barmode="stack",
                     title="Recommended order quantity by store (units)")
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        risk_counts = om["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Lines"]
        fig = px.bar(risk_counts, x="Risk", y="Lines", color="Risk", title="Order lines by risk status")
        st.plotly_chart(fig, use_container_width=True)

    # ---- 3. orders on the way
    st.subheader("3 · Orders on the way")
    if LIVE_MODE:
        pend = []
        for (stn, skn), c in sim_state["combos"].items():
            for arr_date, qty in sorted(c["pending_orders"].items()):
                if qty > 0:
                    pend.append({"Store": stn, "SKU": skn, "Product": PRODUCT_NAME[skn], "Quantity": int(qty),
                                 "Arrives": str(pd.Timestamp(arr_date).date()),
                                 "Days to go": (pd.Timestamp(arr_date) - AS_OF_DATE).days})
        if pend:
            st.dataframe(pd.DataFrame(pend).sort_values(["Days to go", "Store"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Nothing on order at the moment.")
    else:
        st.caption("Static view: the dataset has no explicit purchase-order queue. Start the 🔴 Live Simulation to track real "
                   "orders as they are placed and arrive.")

    # ---- 4. placed orders
    st.subheader("4 · Orders placed this session")
    if st.session_state.placed_orders:
        st.dataframe(pd.DataFrame(st.session_state.placed_orders), use_container_width=True, hide_index=True)
    else:
        st.caption("No orders placed yet.")

    # ---- 5. everything
    with st.expander("Full calculation — every store × SKU"):
        cols = ["Store", "Product", "SKU", "Forecast_During_Lead_Time", "Safety_Stock", "Safety_Stock_Rate",
                "Current_Stock", "Incoming_Stock_Estimate", "Raw_Calculation", "Recommended_Order", "Risk"]
        st.dataframe(
            full_table[cols].rename(columns={
                "Forecast_During_Lead_Time": "Forecast During Lead Time", "Safety_Stock": "Safety Stock",
                "Safety_Stock_Rate": "Safety Stock Rate", "Current_Stock": "Current Stock",
                "Incoming_Stock_Estimate": "Incoming Stock (est.)", "Raw_Calculation": "Raw Calculation",
                "Recommended_Order": "Recommended Order",
            }),
            use_container_width=True, hide_index=True,
        )
    st.info("The system recommends an order because available inventory is below the expected "
            "demand requirement during the replenishment period.")

# ============================================================== 8. RISK DASHBOARD
with tab_risk:
    st.header("Risk Dashboard")
    st.caption(f"Dynamically calculated for all 6 stores × 3 SKUs — as of {AS_OF_DATE.date()} · "
               f"forecast period = {forecast_period} days.")
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        wastage_by_sku = full_table.groupby("Product")["Wastage_Last30d"].sum().reset_index()
        fig = px.bar(wastage_by_sku, x="Product", y="Wastage_Last30d", color="Product",
                     title="Wastage by SKU (last 30d, current scope)")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        stockout_by_sku = full_table.groupby("Product")["Stockout_Last30d"].sum().reset_index()
        fig = px.bar(stockout_by_sku, x="Product", y="Stockout_Last30d", color="Product",
                     title="Stockout Occurrences by SKU (last 30d, current scope)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== LIVE SIMULATION
with tab_live:
    st.header("🔴 Live Simulation")
    st.caption("Simulates ongoing grocery operations day-by-day across the full 6-store × 3-SKU "
               "network, adaptively learning each combo's safety-stock rate from real stockout/overstock outcomes.")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("▶️ Next Day"):
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()
    st.session_state.sim_autoplay = c2.toggle("Auto-play", value=st.session_state.sim_autoplay)
    st.session_state.sim_speed = c3.select_slider("Days per tick", options=[1, 3, 7], value=st.session_state.sim_speed)
    if c4.button("🔄 Reset to historical data"):
        st.session_state.sim_state = simulation.init_simulation_state(
            HISTORICAL_DF, STORES, PRODUCTS, HISTORICAL_AS_OF_DATE, seed=7, actuals_df=TEST_DF)
        st.rerun()

    st.caption(
        f"The simulation starts right after the 70% file, on **{(HISTORICAL_AS_OF_DATE + pd.Timedelta(days=1)).date()}**. "
        f"Until {TEST_DF['Date'].max().date()} it replays the real demand from the held-out 30% test file day by day "
        "(stock decisions are still the simulation's own); after that it keeps going on its learned demand profile."
    )
    st.metric("Days simulated", sim_state["days_simulated"])
    st.metric("Current simulated date", str(AS_OF_DATE.date()))

    st.divider()
    st.subheader("📈 Is it actually learning? — stockout rate, start vs now")
    learning_log = sim_state["learning_log"]
    sim_log = sim_state["log"]
    if learning_log.empty:
        st.info("Run a few simulated days to see whether the adaptive safety-stock rate is working.")
    else:
        n_days = sim_state["days_simulated"]
        window = min(30, max(1, n_days // 4))  # compare first vs last quarter (capped at 30 days)
        first_dates = sorted(sim_log["Date"].unique())[:window]
        last_dates = sorted(sim_log["Date"].unique())[-window:]
        start_rate = sim_log[sim_log["Date"].isin(first_dates)]["Stockout"].mean()
        now_rate = sim_log[sim_log["Date"].isin(last_dates)]["Stockout"].mean()
        delta = now_rate - start_rate

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Stockout rate — first {window} simulated days (fixed 20% baseline)", f"{start_rate:.1%}")
        m2.metric(f"Stockout rate — most recent {window} days (adaptive rate)", f"{now_rate:.1%}",
                  delta=f"{delta:+.1%}", delta_color="inverse")
        rates_now = simulation.current_safety_stock_rates(sim_state)
        avg_rate = float(np.mean(list(rates_now.values()))) if rates_now else 0.20
        m3.metric("Avg. adaptive safety-stock rate now", f"{avg_rate:.0%}", delta=f"{avg_rate-0.20:+.0%} vs 20% baseline")

        if delta < 0:
            st.success(f"The adaptive policy is working: the network stockout rate fell by "
                       f"{abs(delta):.1%} (from {start_rate:.1%} to {now_rate:.1%}) as the model raised "
                       "safety-stock rates on combos that actually stocked out.")
        elif n_days < 14:
            st.info("Not enough simulated days yet to draw a conclusion — run more days.")
        else:
            st.warning("The stockout rate has not clearly improved yet on this run — worth investigating further.")

        st.caption(
            "Trade-off to watch: pushing the safety-stock rate up to kill every stockout also raises "
            "the overstock threshold, so a long-running simulation with mostly stockout-driven "
            "adjustments can end up with more Overstock Risk than it started with. That's a real "
            "trade-off the model is making, not a bug."
        )

        st.subheader("Network Stockout Rate over simulated time")
        rolling = simulation.rolling_stockout_rate(sim_log)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Daily_Stockout_Rate"], name="Daily", opacity=0.4))
        fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Rolling_Stockout_Rate"], name="7-day rolling avg"))
        fig.update_layout(title="Network Stockout Rate — Daily vs 7-day Rolling Average", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Adaptively learned safety-stock rates (current)")
        rate_df = pd.DataFrame([
            {"Store": s, "SKU": k, "Product": PRODUCT_NAME[k], "Safety Stock Rate": f"{r:.1%}"}
            for (s, k), r in rates_now.items()
        ]).sort_values("Safety Stock Rate", ascending=False)
        st.dataframe(rate_df, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download learning log (CSV)", learning_log.to_csv(index=False),
                            file_name="learning_log.csv", mime="text/csv")

    if st.session_state.sim_autoplay:
        for _ in range(st.session_state.sim_speed):
            st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()

# ============================================================== SOLUTION ARCHITECTURE
with tab_arch:
    st.header("Solution Architecture")
    st.markdown(
        "**Data & processing pipeline:**\n\n"
        "```\n"
        "Two Excel files: 70% training (2023-01 -> 2025-04) + 30% test (2025-05 -> 2026-04)\n"
        "    ↓\n"
        "Data Cleaning & Preparation   (pandas: date conversion, type correction, missing-value handling, validation)\n"
        "    ↓\n"
        "Product Movement Analysis     (average daily Units_Sold per SKU -> Fast/Moderate/Slow classification)\n"
        "    ↓\n"
        "Demand Forecasting            (7-day moving-average baseline; Random Forest ML model, chronological 70/30 split)\n"
        "    ↓\n"
        "Inventory Analysis            (Opening/Incoming/Sold/Closing/Wastage/Stockout monitoring)\n"
        "    ↓\n"
        "Lead-Time & Expiry Tracking   (supplier lead time, forecasted demand during lead time, shelf-life/expiry)\n"
        "    ↓\n"
        "Risk Detection                (Normal / Stockout Risk / Overstock Risk / Expiry Risk classification)\n"
        "    ↓\n"
        "Demand-Based Reorder Calculation  (Forecast during lead time + adaptive safety stock - current - incoming)\n"
        "    ↓\n"
        "Live Simulation                (day-by-day network simulation, adaptive safety-stock learning)\n"
        "    ↓\n"
        "Streamlit Dashboard            (Overview, Movement, Forecasting, ML Forecast, Inventory, Stock Intake, "
        "Lead-Time, Expiry, Reorder, Risk, Live Simulation views)\n"
        "```"
    )
    st.markdown(
        "**Technology stack:** Python, Pandas (data handling), scikit-learn (Random Forest forecasting model), "
        "Streamlit (interactive dashboard UI), Plotly (interactive charts)."
    )
    st.markdown(
        "**Sprint 3 upgrade, addressing Sprint 2 lecturer feedback:**\n"
        "1. Two **separate files**: the model trains only on the 70% training file (2023-01 → 2025-04) and is "
        "tested on the separate 30% file (2025-05 → 2026-04) — chronological, never shuffled.\n"
        "2. A **tracking signal** (RSFE ÷ cumulative MAD) is calculated on the 30% test file — and keeps updating through the Live Simulation — to detect whether "
        "the model is systematically over/under-forecasting, not just how large its average error is.\n"
        "3. The **Order Management Dashboard** (Reorder Recommendation tab) shows the live reorder calculation "
        "for every store × SKU combination.\n"
        "4. A bias-correction attempt is reported honestly — including when it does not clearly work."
    )
