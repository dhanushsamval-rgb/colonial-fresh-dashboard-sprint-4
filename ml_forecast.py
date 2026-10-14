"""
Machine-learning demand forecast — Random Forest, network-wide, per SKU.

Train / test design (per lecturer feedback):
  - The model learns ONLY from the separate 70% training file
    (2023-01-01 -> 2025-04-30).
  - It is tested on the separate, held-out 30% test file
    (2025-05-01 -> 2026-04-30) — chronological, never shuffled, so no future
    information leaks into training.
  - Tracking signal (RSFE / cumulative MAD) shows whether the forecast "goes
    around the actual" (stays centred on zero) or drifts one way (the model is
    systematically over- or under-estimating). It is calculated over the 30%
    test window AND keeps updating through the Live Simulation after that.

Forecast upgrade (every choice made using ONLY the 70% training file — the
30% test file is never used to pick anything):
  1. Regularised Random Forest (min_samples_leaf / max_features chosen on a
     validation slice at the end of the 70%), then refit on the FULL 70%.
  2. Adaptive bias correction: when the tracking signal shows the model
     drifting one way, the next forecast is nudged back by an exponentially
     smoothed average of its recent errors. The smoothing rate is also chosen
     on the validation slice: the smallest rate that keeps the tracking signal
     inside the +/-4 limit on at least 90% of validation days.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

TRAIN_SPLIT_RATIO = 0.70
TEST_SPLIT_RATIO = 0.30
TRACKING_SIGNAL_LIMIT = 4.0
N_ESTIMATORS = 200
RANDOM_STATE = 42

# Upgraded Random Forest settings (picked on the validation slice of the 70%)
MIN_SAMPLES_LEAF = 15
MAX_FEATURES = 0.6
VALIDATION_FRACTION = 0.15            # last 15% of the 70% -> validation only
ALPHA_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.30)
TARGET_VALIDATION_BREACH_RATE = 0.10  # want |signal| <= 4 on >= 90% of validation days

FEATURE_COLS = [
    "lag_1", "lag_7", "roll_7_mean", "roll_14_mean", "roll_7_std",
    "dayofweek", "is_weekend", "Promotion_Rate",
]

# Sprint 4 — second ML model: Gradient Boosting (trees built one after another, each
# correcting the previous ones' errors). Same features, same 70/30 split, same adaptive
# bias correction as the Random Forest, so the comparison is fair. Its settings are picked
# on the validation slice at the end of the 70% (lowest validation MAE) — never on the 30%.
GB_GRID = [
    {"n_estimators": n, "learning_rate": lr, "max_depth": d, "min_samples_leaf": leaf}
    for n in (150, 300) for lr in (0.03, 0.06) for d in (2, 3) for leaf in (10, 20)
]
GB_SUBSAMPLE = 0.8
MODEL_LABELS = {"rf": "Random Forest", "gb": "Gradient Boosting"}


def _network_daily_series(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    d = df[df["Product_ID"] == sku].groupby("Date").agg(
        Units_Sold=("Units_Sold", "sum"),
        Promotion_Rate=("Promotion", "mean"),
    ).reset_index().sort_values("Date")
    return d


def _engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["lag_1"] = d["Units_Sold"].shift(1)
    d["lag_7"] = d["Units_Sold"].shift(7)
    d["roll_7_mean"] = d["Units_Sold"].shift(1).rolling(7).mean()
    d["roll_14_mean"] = d["Units_Sold"].shift(1).rolling(14).mean()
    d["roll_7_std"] = d["Units_Sold"].shift(1).rolling(7).std()
    d["dayofweek"] = d["Date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d = d.dropna().reset_index(drop=True)
    return d


def _baseline_predict(d: pd.DataFrame) -> pd.Series:
    """7-day moving-average baseline evaluated on the same rows as the ML test set."""
    return d["roll_7_mean"]


def _metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _tracking_signal(actual, pred) -> np.ndarray:
    """Running Sum of Forecast Errors / cumulative Mean Absolute Deviation.
    Detects whether a model is systematically over- or under-forecasting,
    which MAE/RMSE alone cannot reveal."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    errors = actual - pred
    rsfe = np.cumsum(errors)
    cum_mad = np.cumsum(np.abs(errors)) / np.arange(1, len(errors) + 1)
    cum_mad_safe = np.where(cum_mad == 0, np.nan, cum_mad)
    return rsfe / cum_mad_safe


def _breaches(ts) -> int:
    return int(np.nansum(np.abs(ts) > TRACKING_SIGNAL_LIMIT))


def _adaptive_correct(raw_pred, actual, alpha: float, b0: float = 0.0):
    """One-step-ahead adaptive bias correction.

    For each day: forecast = raw model forecast + current bias estimate.
    Only AFTER the day is over, the bias estimate is updated with that day's
    raw error (actual - raw forecast). Nothing about a day is used to forecast
    that same day. Returns (corrected_forecasts, final_bias_estimate)."""
    b = float(b0)
    out = np.empty(len(raw_pred), dtype=float)
    for i, (r, a) in enumerate(zip(raw_pred, actual)):
        out[i] = max(0.0, float(r) + b)
        b = (1.0 - alpha) * b + alpha * (float(a) - float(r))
    return out, b


def _choose_alpha(raw_val_pred, val_actual):
    """Smallest smoothing rate whose validation tracking signal stays inside
    +/-4 on at least 90% of days (so the correction is as gentle as possible).
    Falls back to the strongest candidate if none reaches that."""
    chosen = ALPHA_CANDIDATES[-1]
    for alpha in ALPHA_CANDIDATES:
        corrected, _ = _adaptive_correct(raw_val_pred, val_actual, alpha)
        ts = _tracking_signal(val_actual, corrected)
        if np.mean(np.abs(ts) > TRACKING_SIGNAL_LIMIT) <= TARGET_VALIDATION_BREACH_RATE:
            chosen = alpha
            break
    return chosen


def _unbiased_reference_breach_pct(residuals, n_perm: int = 300) -> float:
    """What % of days would a PERFECTLY UNBIASED forecaster with this much
    day-to-day noise still spend outside +/-4? (Shuffled, mean-removed errors.)
    A cumulative RSFE/MAD wanders like a random walk over long windows, so
    this is the fair yardstick for the raw tracking signal."""
    rng = np.random.default_rng(RANDOM_STATE)
    e = np.asarray(residuals, dtype=float)
    e = e - e.mean()
    rates = []
    for _ in range(n_perm):
        x = rng.permutation(e)
        ts = np.cumsum(x) / (np.cumsum(np.abs(x)) / np.arange(1, len(x) + 1))
        rates.append(np.mean(np.abs(ts) > TRACKING_SIGNAL_LIMIT))
    return float(np.mean(rates) * 100)


def _make_gb(params: dict) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(subsample=GB_SUBSAMPLE, random_state=RANDOM_STATE, **params)


def _choose_gb_params(fit_train: pd.DataFrame, val: pd.DataFrame):
    """Pick the Gradient Boosting settings with the lowest MAE on the validation slice
    (the last 15% of the 70% file). The 30% test file is never touched here."""
    best_params, best_mae = None, np.inf
    y_val = val["Units_Sold"].values
    for params in GB_GRID:
        m = _make_gb(params).fit(fit_train[FEATURE_COLS], fit_train["Units_Sold"])
        mae = float(np.mean(np.abs(y_val - m.predict(val[FEATURE_COLS]))))
        if mae < best_mae:
            best_params, best_mae = params, mae
    return best_params, best_mae


def _bias_summary(actual, pred) -> dict:
    """Is the forecast error random around zero, or does it lean one way?

    mean error  : > 0 = forecast ran low (under-forecast), < 0 = ran high (over-forecast)
    t statistic : mean error / its standard error. |t| < 2 means the mean error is
                  statistically indistinguishable from zero (no systematic bias).
    lag-1 autocorrelation : near 0 = errors look random day to day; a large positive value
                  means errors come in streaks (a trend the model is missing)."""
    e = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    n = len(e)
    mean = float(np.mean(e))
    sd = float(np.std(e, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and se > 0 else float("nan")
    ac1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if n > 2 else float("nan")
    return {"n": n, "mean_error": mean, "std_error": sd, "t_stat": float(t), "autocorr_lag1": ac1,
            "unbiased": bool(abs(t) < 2) if not np.isnan(t) else False}


def _model_parts(bundle: dict, which: str):
    """(model, alpha, bias_init) for 'rf' or 'gb'."""
    if which == "gb":
        return bundle["gb_model"], bundle["gb_alpha"], bundle["gb_bias_init"]
    return bundle["model"], bundle["alpha"], bundle["bias_init"]


def train_and_backtest(train_df: pd.DataFrame, sku: str, test_df: pd.DataFrame) -> dict:
    """Learn from the 70% training file only; evaluate on the separate 30% test file.

    Lag features for the first test days naturally look back into the end of
    the training period (one-step-ahead evaluation) — that is real, already
    known history, not leakage. The model itself never sees a test row."""
    daily = pd.concat([_network_daily_series(train_df, sku), _network_daily_series(test_df, sku)],
                      ignore_index=True).sort_values("Date")
    feat = _engineer_features(daily)
    train_end = train_df["Date"].max()
    train = feat[feat["Date"] <= train_end]
    test = feat[feat["Date"] > train_end]
    n_total = len(feat)

    # Validation slice = last 15% of the 70%. Used ONLY to choose the bias-correction rate.
    fit_split = int(round(len(train) * (1 - VALIDATION_FRACTION)))
    fit_train, val = train.iloc[:fit_split], train.iloc[fit_split:]

    X_test, y_test = test[FEATURE_COLS], test["Units_Sold"]

    # --- Original Sprint-3 Random Forest (unregularised, 85% of the 70%) — kept for before/after comparison
    rf_original = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_original.fit(fit_train[FEATURE_COLS], fit_train["Units_Sold"])
    orig_pred_test = rf_original.predict(X_test)

    # --- Upgrade step 1: regularised RF; first fit on 85% to choose the correction rate on the validation slice
    def _make_rf():
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
                                     max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=-1)

    val_model = _make_rf().fit(fit_train[FEATURE_COLS], fit_train["Units_Sold"])
    val_raw = val_model.predict(val[FEATURE_COLS])
    val_actual = val["Units_Sold"].values
    alpha = _choose_alpha(val_raw, val_actual)
    _, bias_init = _adaptive_correct(val_raw, val_actual, alpha)   # bias estimate going into the test period

    # --- ...then refit on the FULL 70% so the model learns from all of it
    model = _make_rf().fit(train[FEATURE_COLS], train["Units_Sold"])
    rf_pred_test = model.predict(X_test)                                        # tuned, no correction
    final_pred_test, bias_end = _adaptive_correct(rf_pred_test, y_test.values, alpha, bias_init)  # + adaptive bias correction
    baseline_pred_test = _baseline_predict(test).values

    # Metrics on the 70% training window itself (in-sample) for the side-by-side view
    train_pred = model.predict(train[FEATURE_COLS])
    rf_metrics_train = _metrics(train["Units_Sold"], train_pred)
    baseline_metrics_train = _metrics(train["Units_Sold"], _baseline_predict(train).values)

    ts_orig = _tracking_signal(y_test, orig_pred_test)
    ts_tuned = _tracking_signal(y_test, rf_pred_test)
    ts_final = _tracking_signal(y_test, final_pred_test)
    ts_base = _tracking_signal(y_test, baseline_pred_test)

    # --- Sprint 4: Gradient Boosting, built exactly like the upgraded Random Forest ------------
    gb_params, gb_val_mae = _choose_gb_params(fit_train, val)
    gb_val_model = _make_gb(gb_params).fit(fit_train[FEATURE_COLS], fit_train["Units_Sold"])
    gb_val_raw = gb_val_model.predict(val[FEATURE_COLS])
    gb_alpha = _choose_alpha(gb_val_raw, val_actual)
    _, gb_bias_init = _adaptive_correct(gb_val_raw, val_actual, gb_alpha)
    gb_model = _make_gb(gb_params).fit(train[FEATURE_COLS], train["Units_Sold"])       # refit on the FULL 70%
    gb_pred_test = gb_model.predict(X_test)                                             # tuned, no correction
    gb_final_pred_test, gb_bias_end = _adaptive_correct(gb_pred_test, y_test.values, gb_alpha, gb_bias_init)
    gb_metrics_train = _metrics(train["Units_Sold"], gb_model.predict(train[FEATURE_COLS]))
    ts_gb = _tracking_signal(y_test, gb_final_pred_test)
    ts_gb_tuned = _tracking_signal(y_test, gb_pred_test)

    # Historical (out-of-sample) check: the last 126 days of the 70%, forecast by models that were
    # fitted only on the days BEFORE them, with the same day-by-day bias correction.
    rf_val_corr, _ = _adaptive_correct(val_raw, val_actual, alpha)
    gb_val_corr, _ = _adaptive_correct(gb_val_raw, val_actual, gb_alpha)

    # Winner = lowest MAE on the 30% test file (the lecturer's comparison: "which one is the minimum").
    winner = "gb" if _metrics(y_test, gb_final_pred_test)["MAE"] < _metrics(y_test, final_pred_test)["MAE"] else "rf"

    feature_importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    mean_err = lambda p: float(np.mean(y_test.values - np.asarray(p)))   # >0 = under-forecast, <0 = over-forecast

    return {
        "sku": sku,
        "n_total": n_total,
        "train_days": len(train),
        "bias_val_days": len(val),
        "test_days": len(test),
        "train_split_ratio": TRAIN_SPLIT_RATIO,
        "train_date_range": (train["Date"].min(), train["Date"].max()),
        "test_date_range": (test["Date"].min(), test["Date"].max()),
        "test_dates": test["Date"].values,
        "test_actual": y_test.values,
        "alpha": alpha,
        "bias_init": bias_init,
        "bias_end": bias_end,

        # ---- FINAL model = regularised RF refit on full 70% + adaptive bias correction
        "rf_pred": final_pred_test,
        "rf_metrics": _metrics(y_test, final_pred_test),
        "rf_tracking_signal": ts_final,
        "rf_ts_breaches": _breaches(ts_final),
        "rf_mean_error": mean_err(final_pred_test),

        # ---- same RF, regularised, but WITHOUT the correction
        "rf_tuned_pred": rf_pred_test,
        "rf_tuned_metrics": _metrics(y_test, rf_pred_test),
        "rf_tuned_tracking_signal": ts_tuned,
        "rf_tuned_ts_breaches": _breaches(ts_tuned),
        "rf_tuned_mean_error": mean_err(rf_pred_test),

        # ---- original Sprint-3 Random Forest (before the upgrade)
        "rf_original_pred": orig_pred_test,
        "rf_original_metrics": _metrics(y_test, orig_pred_test),
        "rf_original_tracking_signal": ts_orig,
        "rf_original_ts_breaches": _breaches(ts_orig),
        "rf_original_mean_error": mean_err(orig_pred_test),

        # ---- 7-day moving-average baseline
        "baseline_pred": baseline_pred_test,
        "baseline_metrics": _metrics(y_test, baseline_pred_test),
        "baseline_tracking_signal": ts_base,
        "baseline_ts_breaches": _breaches(ts_base),
        "baseline_mean_error": mean_err(baseline_pred_test),

        "rf_metrics_train": rf_metrics_train,
        "baseline_metrics_train": baseline_metrics_train,
        "unbiased_reference_breach_pct": _unbiased_reference_breach_pct(y_test.values - rf_pred_test),
        "feature_importances": feature_importances,

        # ---- Sprint 4: Gradient Boosting (FINAL = tuned + the same adaptive bias correction)
        "gb_params": gb_params,
        "gb_val_mae": gb_val_mae,
        "gb_alpha": gb_alpha,
        "gb_bias_init": gb_bias_init,
        "gb_bias_end": gb_bias_end,
        "gb_pred": gb_final_pred_test,
        "gb_metrics": _metrics(y_test, gb_final_pred_test),
        "gb_tracking_signal": ts_gb,
        "gb_ts_breaches": _breaches(ts_gb),
        "gb_mean_error": mean_err(gb_final_pred_test),
        "gb_tuned_pred": gb_pred_test,
        "gb_tuned_metrics": _metrics(y_test, gb_pred_test),
        "gb_tuned_tracking_signal": ts_gb_tuned,
        "gb_tuned_ts_breaches": _breaches(ts_gb_tuned),
        "gb_tuned_mean_error": mean_err(gb_pred_test),
        "gb_metrics_train": gb_metrics_train,
        "gb_feature_importances": dict(zip(FEATURE_COLS, gb_model.feature_importances_)),
        "gb_unbiased_reference_breach_pct": _unbiased_reference_breach_pct(y_test.values - gb_pred_test),
        "gb_model": gb_model,

        # ---- winner (lowest MAE on the 30% test file) and the bias checks
        "winner": winner,
        "bias_test": {"rf": _bias_summary(y_test.values, final_pred_test),
                      "gb": _bias_summary(y_test.values, gb_final_pred_test),
                      "baseline": _bias_summary(y_test.values, baseline_pred_test)},
        "history": {                                   # out-of-sample historical window (end of the 70%)
            "dates": val["Date"].values, "actual": val_actual,
            "rf_pred": rf_val_corr, "gb_pred": gb_val_corr,
            "rf_ts": _tracking_signal(val_actual, rf_val_corr),
            "gb_ts": _tracking_signal(val_actual, gb_val_corr),
            "rf_metrics": _metrics(val_actual, rf_val_corr), "gb_metrics": _metrics(val_actual, gb_val_corr),
            "rf_bias": _bias_summary(val_actual, rf_val_corr), "gb_bias": _bias_summary(val_actual, gb_val_corr),
        },
        "model": model,
        "full_features": feat,
        "train_features": train,
    }


def live_tracking_signal(extended_df: pd.DataFrame, sku: str, bundle: dict, live_start) -> dict:
    """Tracking signal over the Live Simulation days (everything after the 70%
    training file). Each simulated day is forecast one step ahead by the model
    trained on the 70% file (with the same adaptive bias correction) and
    compared with what actually sold. RSFE / cumulative MAD keeps accumulating
    for as long as the simulation runs — including after the 30% test window ends."""
    daily = _network_daily_series(extended_df, sku)
    feat = _engineer_features(daily)
    live = feat[feat["Date"] >= pd.Timestamp(live_start)]
    if live.empty:
        return {"dates": np.array([]), "rf": np.array([]), "gb": np.array([]),
                "baseline": np.array([]), "actual": np.array([])}
    actual = live["Units_Sold"].values
    out = {"dates": live["Date"].values, "actual": actual}
    for which in ("rf", "gb"):
        model, alpha, bias_init = _model_parts(bundle, which)
        corrected, _ = _adaptive_correct(model.predict(live[FEATURE_COLS]), actual, alpha, bias_init)
        out[which] = _tracking_signal(actual, corrected)
    out["baseline"] = _tracking_signal(actual, _baseline_predict(live).values)
    return out


def forecast_forward(df: pd.DataFrame, sku: str, model_bundle: dict, as_of_date, forecast_days: int,
                     which: str = "rf") -> pd.DataFrame:
    """Recursive multi-step forecast from as_of_date forward, with the current
    adaptive bias estimate applied. `df` is whatever timeline is current
    (70% file, or 70% + live-simulated days). `which` = 'rf' or 'gb'."""
    model, alpha, bias_init = _model_parts(model_bundle, which)
    # Build features from the CURRENT timeline (never the held-out 30% file)
    feat = _engineer_features(_network_daily_series(df, sku))

    history = feat[feat["Date"] <= as_of_date].copy()
    if history.empty:
        history = feat.copy()

    # Current bias estimate: start from the one learned at the end of validation and
    # walk it through any days after the 70% file (i.e. the live-simulated days).
    train_end = model_bundle["train_date_range"][1]
    bias = float(bias_init)
    live_part = history[history["Date"] > train_end]
    if len(live_part):
        _, bias = _adaptive_correct(model.predict(live_part[FEATURE_COLS]), live_part["Units_Sold"].values,
                                    alpha, bias)

    series = list(history["Units_Sold"].values)
    promo_rate = float(history["Promotion_Rate"].tail(14).mean()) if len(history) else 0.0

    preds = []
    cur_date = pd.Timestamp(as_of_date)
    for _ in range(forecast_days):
        cur_date = cur_date + pd.Timedelta(days=1)
        lag_1 = series[-1]
        lag_7 = series[-7] if len(series) >= 7 else series[0]
        roll_7 = np.mean(series[-7:]) if len(series) >= 7 else np.mean(series)
        roll_14 = np.mean(series[-14:]) if len(series) >= 14 else np.mean(series)
        roll_7_std = np.std(series[-7:]) if len(series) >= 7 else 0.0
        dow = cur_date.dayofweek
        is_weekend = int(dow >= 5)

        X = pd.DataFrame([{
            "lag_1": lag_1, "lag_7": lag_7, "roll_7_mean": roll_7,
            "roll_14_mean": roll_14, "roll_7_std": roll_7_std,
            "dayofweek": dow, "is_weekend": is_weekend, "Promotion_Rate": promo_rate,
        }])[FEATURE_COLS]

        pred = max(0.0, float(model.predict(X)[0]) + bias)
        preds.append({"Date": cur_date, "ML_Forecast": pred})
        series.append(pred)

    return pd.DataFrame(preds)
