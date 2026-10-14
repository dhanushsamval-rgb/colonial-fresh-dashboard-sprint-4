"""Baseline demand forecasting (non-ML) — 7-day moving average projected
forward as a flat daily rate, plus an optional linear-trend comparison."""
import numpy as np
import pandas as pd

MOVING_AVERAGE_WINDOW = 7


def baseline_moving_average_forecast(daily_units_sold: pd.Series, forecast_days: int) -> float:
    """Return a single flat daily forecast value using the trailing
    MOVING_AVERAGE_WINDOW days of actual Units_Sold (never Simulated_True_Demand,
    to avoid data leakage)."""
    if len(daily_units_sold) == 0:
        return 0.0
    window = daily_units_sold.tail(MOVING_AVERAGE_WINDOW)
    return float(window.mean()) if len(window) > 0 else 0.0


def linear_trend_forecast(daily_units_sold: pd.Series, forecast_days: int) -> np.ndarray:
    """Optional scikit-learn-free linear trend line for comparison only —
    not claimed to be superior to the baseline."""
    y = daily_units_sold.values.astype(float)
    n = len(y)
    if n < 2:
        return np.repeat(y[-1] if n else 0.0, forecast_days)
    x = np.arange(n)
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(n, n + forecast_days)
    preds = np.polyval(coeffs, future_x)
    return np.clip(preds, 0, None)
