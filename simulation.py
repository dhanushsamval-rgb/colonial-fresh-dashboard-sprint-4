"""Live Simulation engine: simulates ongoing grocery operations day-by-day
across the full 6-store x 3-SKU network, with adaptive safety-stock learning
based on real stockout/overstock outcomes."""
import numpy as np
import pandas as pd
from datetime import timedelta

from inventory_analysis import calculate_reorder, classify_risk, SAFETY_STOCK_RATE
from forecasting import baseline_moving_average_forecast, MOVING_AVERAGE_WINDOW

RECENT_SALES_BUFFER = 14
DAY_TO_DAY_NOISE_STD = 0.15
WEEKDAY_FACTOR_CLIP = (0.5, 1.6)
PROMO_BOOST_CLIP = (1.0, 2.0)
DEFAULT_PROMO_BOOST = 1.15
LEAD_TIME_VARIABILITY_RATE = 0.3
SAFETY_RATE_MIN = 0.05
SAFETY_RATE_MAX = 5.00
SAFETY_RATE_UP_MULTIPLIER = 1.35
SAFETY_RATE_DOWN_MULTIPLIER = 0.97
OVERSTOCK_COMFORT_MULTIPLIER = 1.5

SIM_LOG_COLUMNS = [
    "Date", "Store", "Product_ID", "Product_Name", "Movement_Class",
    "Opening_Stock", "Incoming_Stock", "Units_Sold", "Closing_Stock",
    "Wastage", "Promotion", "Stockout", "Lead_Time_Days", "Shelf_Life_Days",
    "Selling_Price_AUD",
]
LEAD_TIME_LOG_COLUMNS = ["Store", "Product_ID", "Order_Date", "Planned_Lead_Time", "Actual_Lead_Time"]
LEARNING_LOG_COLUMNS = ["Date", "Store", "Product_ID", "Stockout", "Overstocked", "Safety_Stock_Rate"]
HISTORICAL_SCHEMA_COLUMNS = [
    "Date", "Store", "Product_ID", "Product_Name", "Movement_Class",
    "Opening_Stock", "Incoming_Stock", "Simulated_True_Demand", "Units_Sold",
    "Closing_Stock", "Wastage", "Promotion", "Selling_Price_AUD", "Stockout",
    "Lead_Time_Days", "Shelf_Life_Days", "Expiry_Date",
]


def _learn_combo_profile(hist: pd.DataFrame) -> dict:
    hist = hist.sort_values("Date")
    hist_recent = hist.tail(180) if len(hist) > 180 else hist

    weekday_factors = {}
    overall_mean = max(hist_recent["Units_Sold"].mean(), 0.1)
    for wd in range(7):
        wd_mean = hist_recent[hist_recent["Date"].dt.dayofweek == wd]["Units_Sold"].mean()
        factor = (wd_mean / overall_mean) if pd.notna(wd_mean) and overall_mean > 0 else 1.0
        weekday_factors[wd] = float(np.clip(factor, *WEEKDAY_FACTOR_CLIP))

    promo_col = hist_recent["Promotion"] if "Promotion" in hist_recent else pd.Series([0])
    promo_prob = float(promo_col.mean()) if len(promo_col) else 0.05

    promo_rows = hist_recent[hist_recent["Promotion"] == 1] if "Promotion" in hist_recent else pd.DataFrame()
    non_promo_rows = hist_recent[hist_recent["Promotion"] == 0] if "Promotion" in hist_recent else hist_recent
    if len(promo_rows) > 0 and len(non_promo_rows) > 0 and non_promo_rows["Units_Sold"].mean() > 0:
        promo_boost = promo_rows["Units_Sold"].mean() / non_promo_rows["Units_Sold"].mean()
        promo_boost = float(np.clip(promo_boost, *PROMO_BOOST_CLIP))
    else:
        promo_boost = DEFAULT_PROMO_BOOST

    wastage_lambda = max(float(hist_recent["Wastage"].mean()), 0.1)

    last_row = hist.iloc[-1]
    return {
        "base_rate": overall_mean,
        "weekday_factors": weekday_factors,
        "promo_probability": promo_prob,
        "promo_boost": promo_boost,
        "wastage_lambda": wastage_lambda,
        "lead_time_days": int(last_row["Lead_Time_Days"]),
        "shelf_life_days": int(last_row["Shelf_Life_Days"]),
        "current_stock": float(last_row["Closing_Stock"]),
        "safety_stock_rate": SAFETY_STOCK_RATE,
        "recent_sales": list(hist["Units_Sold"].tail(RECENT_SALES_BUFFER).values),
        "product_name": last_row["Product_Name"],
        "movement_class": last_row["Movement_Class"],
        "selling_price": float(last_row["Selling_Price_AUD"]),
        "pending_orders": {},
    }


def _build_actuals(actuals_df) -> dict:
    """Held-out 30% test data, keyed (date, store, sku) -> (true_demand, promo).
    While the simulation is inside the 30% window it replays this real
    demand day by day; after the window ends it falls back to its learned
    demand profile so it can keep running."""
    if actuals_df is None or len(actuals_df) == 0:
        return {}
    keys = zip(actuals_df["Date"], actuals_df["Store"], actuals_df["Product_ID"])
    vals = zip(actuals_df["Simulated_True_Demand"].astype(float), actuals_df["Promotion"].astype(int))
    return dict(zip(keys, vals))


def init_simulation_state(df: pd.DataFrame, stores, products, as_of_date, seed=None, actuals_df=None):
    rng = np.random.default_rng(seed)
    combos = {}
    for store in stores:
        for sku in products:
            hist = df[(df["Store"] == store) & (df["Product_ID"] == sku) & (df["Date"] <= as_of_date)]
            if hist.empty:
                continue
            combos[(store, sku)] = _learn_combo_profile(hist)

    return {
        "sim_date": pd.Timestamp(as_of_date),
        "combos": combos,
        "log": pd.DataFrame(columns=SIM_LOG_COLUMNS),
        "lead_time_log": pd.DataFrame(columns=LEAD_TIME_LOG_COLUMNS),
        "learning_log": pd.DataFrame(columns=LEARNING_LOG_COLUMNS),
        "rng": rng,
        "days_simulated": 0,
        "actuals": _build_actuals(actuals_df),
    }


def simulate_next_day(state: dict) -> dict:
    rng = state["rng"]
    new_date = state["sim_date"] + timedelta(days=1)
    log_rows = []
    lead_time_rows = []
    learning_rows = []

    for (store, sku), c in state["combos"].items():
        actual = state.get("actuals", {}).get((new_date, store, sku))
        if actual is not None:
            # Inside the 30% test window: replay the real held-out demand
            true_demand, is_promo = float(actual[0]), bool(actual[1])
        else:
            wd = new_date.dayofweek
            wf = c["weekday_factors"].get(wd, 1.0)
            is_promo = rng.random() < c["promo_probability"]
            promo_mult = c["promo_boost"] if is_promo else 1.0
            noise = max(0.05, rng.normal(1.0, DAY_TO_DAY_NOISE_STD))
            true_demand = c["base_rate"] * wf * promo_mult * noise

        arriving = int(c["pending_orders"].pop(new_date, 0))
        opening_stock = c["current_stock"]
        available = opening_stock + arriving

        units_sold = int(min(available, round(true_demand)))
        stockout = 1 if available < true_demand else 0
        after_sales = available - units_sold
        wastage = int(min(after_sales, rng.poisson(c["wastage_lambda"])))
        closing_stock = after_sales - wastage

        c["current_stock"] = closing_stock
        c["recent_sales"].append(units_sold)
        c["recent_sales"] = c["recent_sales"][-RECENT_SALES_BUFFER:]

        avg_daily_forecast = float(np.mean(c["recent_sales"])) if c["recent_sales"] else c["base_rate"]
        forecast_during_lead_time = avg_daily_forecast * c["lead_time_days"]

        incoming_pipeline = sum(c["pending_orders"].values())
        recommended_order, safety_stock, _ = calculate_reorder(
            forecast_during_lead_time, closing_stock, incoming_pipeline, c["safety_stock_rate"]
        )

        if recommended_order > 0:
            planned_lead_time = c["lead_time_days"]
            variability = max(0, round(rng.normal(0, planned_lead_time * LEAD_TIME_VARIABILITY_RATE)))
            actual_lead_time = max(1, planned_lead_time + variability)
            arrival_date = new_date + timedelta(days=actual_lead_time)
            c["pending_orders"][arrival_date] = c["pending_orders"].get(arrival_date, 0) + recommended_order
            lead_time_rows.append({
                "Store": store, "Product_ID": sku, "Order_Date": new_date,
                "Planned_Lead_Time": planned_lead_time, "Actual_Lead_Time": actual_lead_time,
            })

        stock_position = closing_stock + sum(c["pending_orders"].values())
        risk = classify_risk(stock_position, forecast_during_lead_time,
                              days_remaining=c["shelf_life_days"])

        overstocked = stock_position > OVERSTOCK_COMFORT_MULTIPLIER * max(forecast_during_lead_time, 0.01)
        if stockout:
            c["safety_stock_rate"] = min(SAFETY_RATE_MAX, c["safety_stock_rate"] * SAFETY_RATE_UP_MULTIPLIER)
        elif overstocked:
            c["safety_stock_rate"] = max(SAFETY_RATE_MIN, c["safety_stock_rate"] * SAFETY_RATE_DOWN_MULTIPLIER)

        log_rows.append({
            "Date": new_date, "Store": store, "Product_ID": sku,
            "Product_Name": c["product_name"], "Movement_Class": c["movement_class"],
            "Opening_Stock": opening_stock, "Incoming_Stock": arriving,
            "Units_Sold": units_sold, "Closing_Stock": closing_stock,
            "Wastage": wastage, "Promotion": "Yes" if is_promo else "No",
            "Stockout": stockout, "Lead_Time_Days": c["lead_time_days"],
            "Shelf_Life_Days": c["shelf_life_days"], "Selling_Price_AUD": c["selling_price"],
        })
        learning_rows.append({
            "Date": new_date, "Store": store, "Product_ID": sku,
            "Stockout": stockout, "Overstocked": int(overstocked),
            "Safety_Stock_Rate": c["safety_stock_rate"],
        })

    state["sim_date"] = new_date
    state["log"] = pd.concat([state["log"], pd.DataFrame(log_rows)], ignore_index=True)
    if lead_time_rows:
        state["lead_time_log"] = pd.concat([state["lead_time_log"], pd.DataFrame(lead_time_rows)], ignore_index=True)
    state["learning_log"] = pd.concat([state["learning_log"], pd.DataFrame(learning_rows)], ignore_index=True)
    state["days_simulated"] += 1
    return state


def rolling_stockout_rate(log: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=["Date", "Daily_Stockout_Rate", "Rolling_Stockout_Rate"])
    daily = log.groupby("Date")["Stockout"].mean().reset_index()
    daily = daily.rename(columns={"Stockout": "Daily_Stockout_Rate"})
    daily["Rolling_Stockout_Rate"] = daily["Daily_Stockout_Rate"].rolling(window, min_periods=1).mean()
    return daily


def current_safety_stock_rates(state: dict) -> dict:
    return {combo: c["safety_stock_rate"] for combo, c in state["combos"].items()}


def current_incoming_pipeline(state: dict) -> dict:
    """Real total quantity currently on order (any arrival date) for each
    store/SKU combo — used to correctly classify risk during Live Mode
    instead of a naive historical average."""
    return {combo: float(sum(c["pending_orders"].values())) for combo, c in state["combos"].items()}


def place_manual_order(state: dict, store, sku, qty: int, lead_time_days: int = None):
    """Queue a manually placed order into the same pending-orders pipeline
    the simulation itself uses, so it arrives, affects future days, and is
    correctly reflected in risk/stock-position calculations immediately."""
    combo = state["combos"].get((store, sku))
    if combo is None or qty <= 0:
        return None
    lt = lead_time_days if lead_time_days is not None else combo["lead_time_days"]
    arrival_date = state["sim_date"] + timedelta(days=lt)
    combo["pending_orders"][arrival_date] = combo["pending_orders"].get(arrival_date, 0) + int(qty)
    return {"arrival_date": arrival_date, "lead_time_days": lt, "qty": int(qty)}


MIN_LEAD_TIME_OBSERVATIONS = 5


def anticipate_lead_time(lead_time_log: pd.DataFrame, store, sku, planned_lead_time):
    """Anticipate how long the supplier will really take for a Store x SKU, from the
    deliveries actually observed. With fewer than MIN_LEAD_TIME_OBSERVATIONS the
    planned lead time is used (not enough evidence to say otherwise)."""
    planned = int(planned_lead_time)
    sub = lead_time_log[(lead_time_log["Store"] == store) & (lead_time_log["Product_ID"] == sku)]
    if sub.empty:
        return {"n_observations": 0, "avg_actual": float(planned), "min_actual": planned,
                "max_actual": planned, "p80_actual": float(planned), "on_time_pct": None,
                "anticipated_days": planned, "basis": "planned (no deliveries observed yet)"}
    actual = sub["Actual_Lead_Time"].astype(float)
    n = len(sub)
    avg = float(actual.mean())
    if n >= MIN_LEAD_TIME_OBSERVATIONS:
        anticipated = max(planned, int(np.floor(avg + 0.5)))
        basis = f"observed average of {n} deliveries"
    else:
        anticipated = planned
        basis = f"planned (only {n} deliveries observed, need {MIN_LEAD_TIME_OBSERVATIONS})"
    return {
        "n_observations": n, "avg_actual": avg,
        "min_actual": float(actual.min()), "max_actual": float(actual.max()),
        "p80_actual": float(np.percentile(actual, 80)),
        "on_time_pct": float((actual <= planned).mean()),
        "anticipated_days": anticipated, "basis": basis,
    }


def extended_dataframe(historical_df: pd.DataFrame, sim_log: pd.DataFrame, products: dict) -> pd.DataFrame:
    if sim_log.empty:
        return historical_df

    sim = sim_log.copy()
    sim["Simulated_True_Demand"] = np.nan
    # Same encoding as the loaded historical data (1/0), so the whole timeline is consistent
    sim["Promotion"] = sim["Promotion"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    sim["Expiry_Date"] = sim["Date"] + pd.to_timedelta(sim["Shelf_Life_Days"].astype(int), unit="D")
    sim = sim[HISTORICAL_SCHEMA_COLUMNS]

    combined = pd.concat([historical_df[HISTORICAL_SCHEMA_COLUMNS], sim], ignore_index=True)
    combined = combined.sort_values(["Product_ID", "Store", "Date"]).reset_index(drop=True)
    return combined
