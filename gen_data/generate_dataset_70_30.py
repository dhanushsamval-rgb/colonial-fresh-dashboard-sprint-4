"""
Colonial Fresh synthetic dataset generator.

One continuous series (2023-01-01 -> 2026-04-30) is generated so stock and
accounting stay consistent across the boundary, then written out as TWO
separate files:

  70% TRAINING file : 2023-01-01 -> 2025-04-30   (the only data the model learns from)
  30% TEST file     : 2025-05-01 -> 2026-04-30   (held out, compared against separately)

The dashboard loads the 70% file as its historical data ("today" = 2025-04-30,
so everything forward starts May 2025). The 30% file is used only to test the
model and to feed the Live Simulation with the demand that "actually happened".

Everything else (stores, SKUs, statistical parameters, accounting-identity
integrity) matches the dataset used in earlier sprints.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

START_DATE = "2023-01-01"
END_DATE = "2026-04-30"
SPLIT_DATE = "2025-04-30"   # last day of the 70% training file

STORES = ["Chadstone", "Doncaster", "Forest Hill", "Glen Waverley", "Malvern", "The Pines"]
STORE_SCALE = {
    "Chadstone": 1.35,
    "Doncaster": 1.00,
    "Forest Hill": 0.85,
    "Glen Waverley": 1.10,
    "Malvern": 0.55,
    "The Pines": 0.95,
}

PRODUCTS = {
    "SKU001": {"name": "Bananas", "base_rate": 14.0, "lead_time": 1, "shelf_life": 5, "price": 0.69},
    "SKU002": {"name": "Apples", "base_rate": 6.0, "lead_time": 2, "shelf_life": 10, "price": 0.99},
    "SKU003": {"name": "Strawberries", "base_rate": 1.5, "lead_time": 2, "shelf_life": 4, "price": 3.49},
}

WEEKDAY_FACTOR = {0: 0.85, 1: 0.85, 2: 0.90, 3: 0.95, 4: 1.10, 5: 1.35, 6: 1.20}
PROMO_PROBABILITY = 0.0758
PROMO_BOOST = 1.266
DEMAND_NOISE_STD = 0.233
WASTAGE_LAMBDA = 1.3

REORDER_POINT_MULT = 1.6
ORDER_QTY_TARGET_MULT = 3.2
INITIAL_STOCK_MULT = 2.3

rng = np.random.default_rng(42)


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def generate_combo(store, sku):
    meta = PRODUCTS[sku]
    scale = STORE_SCALE[store]
    base_rate = meta["base_rate"] * scale
    lead_time = meta["lead_time"]
    shelf_life = meta["shelf_life"]
    price = meta["price"]

    reorder_point = int(round(base_rate * lead_time * REORDER_POINT_MULT))
    order_qty_target = int(round(base_rate * lead_time * ORDER_QTY_TARGET_MULT))
    current_stock = int(round(base_rate * lead_time * INITIAL_STOCK_MULT))

    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)

    pending_orders = {}  # arrival_date -> qty
    rows = []

    for d in daterange(start, end):
        wd = d.weekday()
        wf = WEEKDAY_FACTOR[wd]
        promo = rng.random() < PROMO_PROBABILITY
        promo_mult = PROMO_BOOST if promo else 1.0
        noise = max(0.05, rng.normal(1.0, DEMAND_NOISE_STD))
        true_demand = base_rate * wf * promo_mult * noise
        simulated_true_demand = true_demand

        arriving = int(pending_orders.pop(d, 0))

        opening_stock = current_stock  # PRE-delivery, matches original convention
        available_for_sale = current_stock + arriving  # POST-delivery, used for sales

        units_sold = int(min(available_for_sale, round(true_demand)))
        stockout = 1 if available_for_sale < true_demand else 0

        after_sales = available_for_sale - units_sold

        wastage = int(min(after_sales, rng.poisson(WASTAGE_LAMBDA)))
        closing_stock = after_sales - wastage

        current_stock = closing_stock

        # reorder logic (simple point-based, on post-sale position incl pipeline)
        pipeline_qty = sum(pending_orders.values())
        if (current_stock + pipeline_qty) <= reorder_point:
            order_qty = order_qty_target
            arrival_date = d + timedelta(days=lead_time)
            pending_orders[arrival_date] = pending_orders.get(arrival_date, 0) + order_qty

        expiry_date = d + timedelta(days=shelf_life)

        rows.append({
            "Date": d,
            "Store": store,
            "Product_ID": sku,
            "Product_Name": meta["name"],
            "Movement_Class": "",  # filled later from calculated averages
            "Opening_Stock": opening_stock,
            "Incoming_Stock": arriving,
            "Simulated_True_Demand": round(simulated_true_demand, 2),
            "Units_Sold": units_sold,
            "Closing_Stock": closing_stock,
            "Wastage": wastage,
            "Promotion": "Yes" if promo else "No",
            "Selling_Price_AUD": price,
            "Stockout": stockout,
            "Lead_Time_Days": lead_time,
            "Shelf_Life_Days": shelf_life,
            "Expiry_Date": expiry_date,
        })

        # accounting identity check
        assert opening_stock + arriving - units_sold - wastage == closing_stock

    return rows


def main():
    all_rows = []
    for store in STORES:
        for sku in PRODUCTS:
            all_rows.extend(generate_combo(store, sku))

    df = pd.DataFrame(all_rows)

    # Split chronologically into the 70% training file and the 30% test file
    split = pd.Timestamp(SPLIT_DATE)
    df["Date"] = pd.to_datetime(df["Date"])
    train_df = df[df["Date"] <= split].copy()
    test_df = df[df["Date"] > split].copy()

    # Movement class is assigned from the 70% TRAINING data only (no peeking at the test period)
    avg_sales = train_df.groupby("Product_ID")["Units_Sold"].mean().sort_values(ascending=False)
    ranked = list(avg_sales.index)
    movement_map = {ranked[0]: "Fast-moving", ranked[1]: "Moderate-moving", ranked[2]: "Slow-moving"}
    train_df["Movement_Class"] = train_df["Product_ID"].map(movement_map)
    test_df["Movement_Class"] = test_df["Product_ID"].map(movement_map)

    out_dir = "/home/claude/colonial_fresh_dashboard/data/"
    outputs = [
        (train_df, out_dir + "Colonial_Fresh_70pct_Training_2023-01_to_2025-04.xlsx", "70% TRAINING data",
         f"{train_df['Date'].min().date()} to {train_df['Date'].max().date()}"),
        (test_df, out_dir + "Colonial_Fresh_30pct_Test_2025-05_to_2026-04.xlsx", "30% TEST data",
         f"{test_df['Date'].min().date()} to {test_df['Date'].max().date()}"),
    ]
    for part, path, label, span in outputs:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            part.to_excel(writer, sheet_name="Mock_Data", index=False)
            pd.DataFrame({"Note": [
                f"Colonial Fresh simulated dataset for MGT5ELS Assessment 3 — {label}.",
                f"Date range: {span}.",
                "Chronological 70/30 split of one continuous simulated series: the 70% file is what the "
                "model trains on; the 30% file is held out for testing and the Live Simulation.",
                "Simulated data only — not real Colonial Fresh data, not connected to any live system.",
            ]}).to_excel(writer, sheet_name="README", index=False)
        print("Wrote:", path, part.shape)
    total = len(train_df) + len(test_df)
    print(f"Split check: train {len(train_df)/total:.1%} / test {len(test_df)/total:.1%} of {total} rows")
    print(avg_sales)
    print(movement_map)


if __name__ == "__main__":
    main()
