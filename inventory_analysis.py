"""Inventory analysis: reorder calculation, risk classification, stock
position assessment, and full snapshot tables driving every dashboard tab."""
import pandas as pd
from datetime import timedelta

SAFETY_STOCK_RATE = 0.20
EXPIRY_RISK_DAYS_THRESHOLD = 2
OVERSTOCK_MULTIPLIER = 2.0
RECENT_WINDOW_DAYS = 30


def calculate_reorder(forecasted_demand_during_lead_time, current_stock, incoming_stock,
                       safety_stock_rate: float = SAFETY_STOCK_RATE):
    """Recommended Order = Forecast During Lead Time + Safety Stock
    - Current Stock - Incoming Stock (floored at 0)."""
    safety_stock = safety_stock_rate * forecasted_demand_during_lead_time
    raw = forecasted_demand_during_lead_time + safety_stock - current_stock - incoming_stock
    recommended = max(0, round(raw))
    return recommended, safety_stock, raw


def classify_risk(current_stock, forecasted_demand_during_lead_time, days_remaining,
                   overstock_threshold_multiplier: float = OVERSTOCK_MULTIPLIER):
    """Priority: Expiry Risk > Stockout Risk > Overstock Risk > Normal."""
    if days_remaining is not None and days_remaining <= EXPIRY_RISK_DAYS_THRESHOLD:
        return "Expiry Risk"
    if current_stock < forecasted_demand_during_lead_time:
        return "Stockout Risk"
    if current_stock > overstock_threshold_multiplier * forecasted_demand_during_lead_time:
        return "Overstock Risk"
    return "Normal"


def assess_stock_position(current_stock, forecast_during_lead_time, safety_stock,
                           overstock_multiplier: float = OVERSTOCK_MULTIPLIER):
    """Plain-language verdict on whether a stock level (hypothetical or actual)
    is understocked, on target, well stocked, or overstocked."""
    reorder_point = forecast_during_lead_time + safety_stock
    overstock_threshold = overstock_multiplier * forecast_during_lead_time

    if forecast_during_lead_time <= 0:
        return {
            "status": "Insufficient Data",
            "message": "Not enough forecast signal for this store/SKU yet.",
            "reorder_point": reorder_point,
            "overstock_threshold": overstock_threshold,
        }

    if current_stock < forecast_during_lead_time:
        status, message = "Understocked", (
            f"Current stock ({current_stock:.0f} units) is below forecasted demand during "
            f"lead time ({forecast_during_lead_time:.0f} units). Stockout risk before the next delivery."
        )
    elif current_stock < reorder_point:
        status, message = "Below Target", (
            f"Current stock ({current_stock:.0f} units) covers forecast demand but is below "
            f"the reorder point ({reorder_point:.0f} units) once safety stock is included."
        )
    elif current_stock <= overstock_threshold:
        status, message = "Well Stocked", (
            f"Current stock ({current_stock:.0f} units) is comfortably between the reorder "
            f"point ({reorder_point:.0f}) and the overstock threshold ({overstock_threshold:.0f})."
        )
    else:
        status, message = "Overstocked", (
            f"Current stock ({current_stock:.0f} units) is well above the overstock threshold "
            f"({overstock_multiplier:.0f}x forecasted demand during the lead time = "
            f"{overstock_threshold:.0f} units). This ties up cash and, for a perishable product, "
            f"raises the chance some of it expires before it sells."
        )

    return {
        "status": status,
        "message": message,
        "reorder_point": reorder_point,
        "overstock_threshold": overstock_threshold,
    }


def assess_order(current_stock, incoming_qty, order_qty, avg_daily_forecast, lead_time_days,
                 safety_rate, shelf_life_days, overstock_multiplier: float = OVERSTOCK_MULTIPLIER):
    """AI check of a manually entered order, using the same demand forecast and
    reorder-point maths as the rest of the app.

    Stock position after the order = current stock + already on order + this order.
      below forecast demand during lead time  -> Understocking (stockout likely)
      below reorder point (forecast + safety) -> Slightly understocking
      between reorder point and 2x forecast   -> Order looks right
      above 2x forecast during lead time      -> Overstocking
    Also flags perishables where the position covers more days than the shelf life."""
    forecast_lt = float(avg_daily_forecast) * float(lead_time_days)
    safety_stock = float(safety_rate) * forecast_lt
    reorder_point = forecast_lt + safety_stock
    overstock_threshold = overstock_multiplier * forecast_lt
    recommended, _, _ = calculate_reorder(forecast_lt, current_stock, incoming_qty, safety_rate)

    position_before = float(current_stock) + float(incoming_qty)
    position_after = position_before + float(order_qty)
    days_cover = position_after / avg_daily_forecast if avg_daily_forecast > 0 else None

    base = assess_stock_position(position_after, forecast_lt, safety_stock, overstock_multiplier)
    status_map = {
        "Understocked": ("Understocking", "error"),
        "Below Target": ("Slightly understocking", "warning"),
        "Well Stocked": ("Order looks right", "success"),
        "Overstocked": ("Overstocking", "warning"),
        "Insufficient Data": ("Not enough data", "info"),
    }
    verdict, level = status_map[base["status"]]

    if order_qty == 0 and recommended == 0 and base["status"] != "Insufficient Data":
        verdict, level = "No order needed", "success"
        extra = (f" It is already above the overstock threshold ({overstock_threshold:.0f}) — do not order more."
                 if base["status"] == "Overstocked" else "")
        explanation = (f"Stock on hand plus what is already on order ({position_before:.0f} units) already covers "
                       f"forecast demand during the lead time ({forecast_lt:.0f}) plus safety stock, "
                       f"so ordering nothing is correct.{extra}")
    elif base["status"] == "Well Stocked":
        explanation = (f"After this order the stock position is {position_after:.0f} units, between the reorder point "
                       f"({reorder_point:.0f}) and the overstock threshold ({overstock_threshold:.0f}).")
    elif base["status"] == "Understocked":
        explanation = (f"After this order the stock position is only {position_after:.0f} units, below the "
                       f"{forecast_lt:.0f} units forecast to sell during the lead time — a stockout is likely "
                       f"before the next delivery.")
    elif base["status"] == "Below Target":
        explanation = (f"After this order the stock position is {position_after:.0f} units: enough for forecast demand "
                       f"({forecast_lt:.0f}) but under the reorder point of {reorder_point:.0f} once safety stock is included.")
    elif base["status"] == "Overstocked":
        explanation = (f"After this order the stock position is {position_after:.0f} units, above the overstock threshold "
                       f"of {overstock_threshold:.0f} ({overstock_multiplier:.0f}x forecast demand during the lead time). "
                       f"That ties up cash and raises the chance of waste.")
    else:
        explanation = base["message"]

    expiry_warning = None
    if days_cover is not None and shelf_life_days and days_cover > shelf_life_days and order_qty > 0:
        expiry_warning = (f"This position is about {days_cover:.1f} days of sales but the product's shelf life is "
                          f"{shelf_life_days} days — some of it may expire before it sells.")

    return {
        "verdict": verdict, "level": level, "explanation": explanation, "expiry_warning": expiry_warning,
        "forecast_during_lead_time": forecast_lt, "safety_stock": safety_stock,
        "reorder_point": reorder_point, "overstock_threshold": overstock_threshold,
        "recommended_order": int(recommended), "order_vs_recommended": int(order_qty) - int(recommended),
        "position_before": position_before, "position_after": position_after, "days_cover": days_cover,
    }


def _combo_snapshot(df, store, sku, as_of_date, forecast_days, stock_adjustment, safety_rate,
                     incoming_pipeline_qty=None):
    from forecasting import baseline_moving_average_forecast

    sub = df[(df["Store"] == store) & (df["Product_ID"] == sku) & (df["Date"] <= as_of_date)]
    if sub.empty:
        return None
    sub = sub.sort_values("Date")
    today_row = sub[sub["Date"] == as_of_date]
    if today_row.empty:
        today_row = sub.tail(1)
    today_row = today_row.iloc[-1]

    product_name = today_row["Product_Name"]
    movement_class = today_row["Movement_Class"]
    lead_time = int(today_row["Lead_Time_Days"])
    shelf_life = int(today_row["Shelf_Life_Days"])

    opening_stock = float(today_row["Opening_Stock"])
    incoming_stock_today = float(today_row["Incoming_Stock"])
    units_sold_today = float(today_row["Units_Sold"])
    closing_stock = float(today_row["Closing_Stock"])
    wastage_today = float(today_row["Wastage"])
    stockout_today = "Yes" if today_row["Stockout"] else "No"

    current_stock = closing_stock + (stock_adjustment or 0)

    daily_sold = sub.set_index("Date")["Units_Sold"]
    avg_daily_forecast = baseline_moving_average_forecast(daily_sold, forecast_days)
    forecast_next_period = avg_daily_forecast * forecast_days
    forecast_during_lead_time = avg_daily_forecast * lead_time

    recent = sub[sub["Date"] > as_of_date - timedelta(days=RECENT_WINDOW_DAYS)]
    wastage_recent = float(recent["Wastage"].sum())
    stockout_recent = int(recent["Stockout"].sum())

    if incoming_pipeline_qty is not None:
        # REAL pending-order pipeline from the Live Simulation (or a manually
        # placed order) — not a guess. Used both for the reorder calc and,
        # critically, for the risk classification below.
        incoming_estimate = float(incoming_pipeline_qty)
    else:
        # No live pipeline tracked (static/historical view): fall back to a
        # simple estimate of incoming stock over the lead-time window (avg
        # received over the last lead-time days), since the raw dataset has
        # no explicit PO queue. This estimate is NOT added to stock position
        # for risk classification below — only real, tracked pipeline is.
        recent_lead = sub.tail(max(lead_time, 1))
        incoming_estimate = float(recent_lead["Incoming_Stock"].mean()) if len(recent_lead) else 0.0

    recommended_order, safety_stock, raw_calc = calculate_reorder(
        forecast_during_lead_time, current_stock, incoming_estimate, safety_rate
    )

    last_incoming_date = sub[sub["Incoming_Stock"] > 0]["Date"].max()
    if pd.notna(last_incoming_date):
        expiry_date = last_incoming_date + timedelta(days=shelf_life)
    else:
        expiry_date = as_of_date + timedelta(days=shelf_life)
    days_remaining = (expiry_date - as_of_date).days

    # Stock position for risk purposes = current stock + REAL incoming
    # pipeline only (never the rough historical average) — this is what
    # fixes the false "Stockout Risk" that fired constantly during Live
    # Simulation before this was tracked properly.
    stock_position = current_stock + (incoming_pipeline_qty if incoming_pipeline_qty is not None else 0.0)
    risk = classify_risk(stock_position, forecast_during_lead_time, days_remaining)

    return {
        "Store": store, "SKU": sku, "Product": product_name, "Movement": movement_class,
        "Opening_Stock": opening_stock, "Incoming_Stock": incoming_stock_today,
        "Units_Sold": units_sold_today, "Closing_Stock": closing_stock,
        "Wastage": wastage_today, "Stockout": stockout_today,
        "Current_Stock": current_stock,
        "Avg_Daily_Forecast": avg_daily_forecast,
        "Forecast_Next_Period": forecast_next_period,
        "Lead_Time_Days": lead_time,
        "Forecast_During_Lead_Time": forecast_during_lead_time,
        "Incoming_Stock_Estimate": incoming_estimate,
        "Safety_Stock": safety_stock,
        "Safety_Stock_Rate": safety_rate,
        "Raw_Calculation": raw_calc,
        "Recommended_Order": recommended_order,
        "Shelf_Life_Days": shelf_life,
        "Expiry_Date": expiry_date,
        "Days_Remaining": days_remaining,
        "Risk": risk,
        "Wastage_Last30d": wastage_recent,
        "Stockout_Last30d": stockout_recent,
    }


def build_full_snapshot_table(df, stores, products, as_of_date, forecast_days,
                               stock_adjustments=None, safety_stock_rates=None,
                               incoming_pipeline=None):
    stock_adjustments = stock_adjustments or {}
    safety_stock_rates = safety_stock_rates or {}
    rows = []
    for store in stores:
        for sku in products:
            adj = stock_adjustments.get((store, sku), 0)
            rate = safety_stock_rates.get((store, sku), SAFETY_STOCK_RATE)
            pipeline_qty = incoming_pipeline.get((store, sku)) if incoming_pipeline is not None else None
            row = _combo_snapshot(df, store, sku, as_of_date, forecast_days, adj, rate, pipeline_qty)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_risk_table(df, stores, products, as_of_date, forecast_days,
                      stock_adjustments=None, safety_stock_rates=None, incoming_pipeline=None):
    full = build_full_snapshot_table(df, stores, products, as_of_date, forecast_days,
                                      stock_adjustments, safety_stock_rates, incoming_pipeline)
    if full.empty:
        return full
    return full[[
        "Store", "SKU", "Product", "Movement", "Current_Stock",
        "Forecast_Next_Period", "Lead_Time_Days", "Days_Remaining",
        "Risk", "Recommended_Order",
    ]].rename(columns={"Forecast_Next_Period": "Forecast", "Days_Remaining": "Expiry_Days_Left"})
