# Colonial Fresh — AI-Based Demand Forecasting & Inventory Management

MGT5ELS Assessment 3 · Sprint 3 — Technical Upgrade

## What changed in this update

1. **Two separate data files — real 70% / 30%.**
   - `data/Colonial_Fresh_70pct_Training_2023-01_to_2025-04.xlsx` — the **70% training data** (2023-01-01 → 2025-04-30). This is the only data the model learns from and the dashboard's "historical" data ("today" = 2025-04-30).
   - `data/Colonial_Fresh_30pct_Test_2025-05_to_2026-04.xlsx` — the **30% test data** (2025-05-01 → 2026-04-30), held out. Exactly 70.0% / 30.0% of all rows.
   Nothing is split inside the app any more; the model trains on file 1 and is tested on file 2.

2. **70% vs 30% shown side by side** in the ML Forecast tab (MAE / RMSE / MAPE for Random Forest and the 7-day baseline on each).

3. **Tracking signal keeps updating after the 30% window.** It is calculated over the 30% test file, then continues day by day through the Live Simulation (past 2026-04-30 too).

4. **Live Simulation starts May 2025** (the day after the 70% file) and replays the real demand from the 30% file until it runs out, then continues on its learned demand profile.

5. **New 📦 Manual Order tab — the AI checks your order before you place it.** Enter a quantity; the AI uses the demand forecast, the anticipated supplier lead time, the reorder point (forecast during lead time + safety stock) and stock already on order to say **Understocking / Slightly understocking / Order looks right / Overstocking / No order needed**, with the numbers behind it, a recommended quantity, and a shelf-life warning for perishables.

6. **🚚 Lead-Time Anticipation is now a supplier delivery forecast.** Per Store × SKU: planned vs anticipated lead time (learned from observed deliveries), worst-likely lead time, on-time %, expected / latest-likely arrival dates, and the extra demand to cover if the supplier is late. The anticipated lead time feeds the Manual Order check.

7. **Forecast upgraded and justified on the 30% test data.** The Random Forest learns only from the 70% file. Improvements were chosen using a validation slice inside the 70% (never the 30%): a regularised forest refit on all of the 70%, plus an **adaptive bias correction** that pulls the forecast back when the tracking signal shows it drifting. The ML tab compares the baseline, the original forest, the upgraded forest and the final model on accuracy (MAE/RMSE/MAPE), average error and tracking signal, with an automatic verdict.

8. **🧾 Order Management Dashboard** (tab 7): KPIs, priority-ranked orders to place now (with anticipated lead time, expected arrival and an AI check), "Place ALL recommended orders", orders on the way, orders placed this session, and the full calculation.

9. Earlier fixes retained: false "Stockout Risk" bug fixed (real incoming-order pipeline used for risk), learning visibility (stockout rate start vs now), Recommended Orders + Place order with lead time in Stock Intake.

## Sprint 3 upgrades (per Sprint 2 lecturer feedback)

1. **Explicit 70/30 chronological split** (`ml_forecast.py`) — never shuffled.
2. **Tracking signal** (RSFE ÷ cumulative MAD, ±4 control limit).
3. **Order Management Dashboard** — Reorder Recommendation tab + the new
   Recommended Orders panel in Stock Intake.
4. **Bias-correction attempt** — reported honestly either way.

## Structure

- `data_utils.py` — load/clean/validate the Excel dataset
- `forecasting.py` — 7-day moving-average baseline forecast
- `inventory_analysis.py` — reorder calculation, risk classification, stock assessment
- `ml_forecast.py` — Random Forest model, 70/30 backtest, tracking signal
- `simulation.py` — Live Simulation engine (day-by-day, adaptive safety-stock learning, real order pipeline)
- `app.py` — Streamlit dashboard (13 tabs, all showing all 6 stores × all 3 SKUs)
- `gen_data/generate_dataset_70_30.py` — synthetic dataset generator (writes both files)
- `data/` — the 70% training file and the 30% test file

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```

Simulated data only — not real Colonial Fresh data, not connected to any live system.
