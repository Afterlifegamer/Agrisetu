"""
evaluate_models.py — Individual Accuracy Check for Both Models
==============================================================
Run from the project root:
    python src/scripts/evaluate_models.py

Evaluates:
  1. XGBoost Crop Suitability Model  → classification report + cross-val accuracy
  2. Prophet Price Prediction Model  → MAE & MAPE per crop (last-3-months holdout)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR  = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
NEW_DATA_DIR = os.path.join(DATA_DIR, "New")
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# 1.  XGBoost Suitability Model Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_suitability_model():
    print("\n" + "="*60)
    print("  MODEL 1: XGBoost Crop Suitability Model")
    print("="*60)

    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import data_utils
    from suitability_model import CropSuitabilityModel

    # Re-build the same training dataset the model was trained on
    sm = CropSuitabilityModel()

    weather_data = sm.weather_data
    crop_data    = sm.crop_data
    climatology  = sm.climatology_calendar
    global_avg   = sm.global_avg_rainfall

    yield_file = os.path.join(DATA_DIR, "kerala_monthly_estimates_2023_2025.csv")
    try:
        yield_df = pd.read_csv(yield_file)
    except Exception as e:
        print(f"  ✗ Could not load yield data: {e}")
        return

    from dateutil.relativedelta import relativedelta
    import datetime

    training_rows = []
    for _, row in crop_data.iterrows():
        crop_name = str(row["crop_name"]).strip()
        duration_days   = data_utils.parse_duration_to_days(row["duration_to_harvest"])
        duration_months = max(1, round(duration_days / 30.0))

        crop_yields = yield_df[yield_df["Crop"].str.lower() == crop_name.lower()]
        if crop_yields.empty:
            continue

        pos_yields = crop_yields[crop_yields["Monthly_Production_MT"] > 0]["Monthly_Production_MT"]
        q33 = pos_yields.quantile(0.33) if not pos_yields.empty else 0
        q66 = pos_yields.quantile(0.66) if not pos_yields.empty else 0

        for _, y_row in crop_yields.iterrows():
            harvest_month = int(y_row["Month"])
            harvest_year  = int(y_row["Year"])
            volume        = float(y_row["Monthly_Production_MT"])

            harvest_date  = datetime.datetime(harvest_year, harvest_month, 1)
            planting_date = harvest_date - relativedelta(months=duration_months)
            price_at_plant = sm._get_historical_price(crop_name, planting_date.year, planting_date.month)

            label = 0 if volume == 0 else (2 if volume >= q66 else (1 if volume >= q33 else 0))

            avg_rain = sm._get_avg_rain_for_window(harvest_month, duration_months)
            training_rows.append({
                "rain_mm":               avg_rain,
                "water_dependency_num":  row["water_dependency_num"],
                "time_effort_num":       row["time_effort_num"],
                "price_at_planting":     price_at_plant,
                "label":                 label,
                "crop":                  crop_name,
            })

    if not training_rows:
        print("  ✗ No training rows could be reconstructed.")
        return

    df_train = pd.DataFrame(training_rows)
    X = df_train[["rain_mm", "water_dependency_num", "time_effort_num", "price_at_planting"]]
    y = df_train["label"]

    print(f"\n  Training samples : {len(df_train)}")
    print(f"  Crops covered    : {df_train['crop'].nunique()}")
    print(f"  Label distribution:")
    for lbl, name in {0: "Low (0)", 1: "Medium (1)", 2: "High (2)"}.items():
        pct = (y == lbl).sum() / len(y) * 100
        print(f"    {name}: {(y == lbl).sum()} samples ({pct:.1f}%)")

    # ── Cross-validated accuracy ─────────────────────────────────────────────
    print("\n  Running 5-fold stratified cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss")
    scores = cross_val_score(model_cv, X, y, cv=cv, scoring="accuracy")
    print(f"\n  Cross-Val Accuracy  : {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  Per-fold scores     : {[f'{s*100:.1f}%' for s in scores]}")

    # ── Train on full set, report on full set (in-sample sanity check) ───────
    model_cv.fit(X, y)
    y_pred = model_cv.predict(X)
    print(f"\n  In-sample Accuracy  : {accuracy_score(y, y_pred)*100:.1f}%")

    print("\n  Classification Report:")
    print(classification_report(y, y_pred,
                                target_names=["Low Suit.", "Med. Suit.", "High Suit."],
                                zero_division=0))

    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y, y_pred)
    print(f"  {'':15} {'Low':>8} {'Med':>8} {'High':>8}")
    for i, row_name in enumerate(["Low (actual)", "Med (actual)", "High (actual)"]):
        print(f"  {row_name:15} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # ── Feature importance ───────────────────────────────────────────────────
    feat_names = ["rain_mm", "water_dependency", "time_effort", "price_at_planting"]
    importances = model_cv.feature_importances_
    print("\n  Feature Importances:")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {name:<22} {imp:.3f}  {bar}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  Prophet Price Prediction Model Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_price_model():
    print("\n" + "="*60)
    print("  MODEL 2: Prophet Price Prediction Model")
    print("  (Holdout: last 3 months of each crop's data)")
    print("="*60)

    from prophet import Prophet

    CROP_FILES = {
        "Arecanut":  os.path.join(NEW_DATA_DIR, "arecanut_price.csv"),
        "Cashew":    os.path.join(NEW_DATA_DIR, "cashew_price_data.csv"),
        "Coconut":   os.path.join(NEW_DATA_DIR, "coconut_price_data.csv"),
        "Coffee":    os.path.join(NEW_DATA_DIR, "coffee_price_data.csv"),
        "Ginger":    os.path.join(NEW_DATA_DIR, "ginger_price_data.csv"),
        "Jackfruit": os.path.join(NEW_DATA_DIR, "jackfruit_price_data.csv"),
        "Paddy":     os.path.join(NEW_DATA_DIR, "paddy_price.csv"),
        "Papaya":    os.path.join(NEW_DATA_DIR, "papaya_price_data.csv"),
        "Pepper":    os.path.join(NEW_DATA_DIR, "pepper_price_data.csv"),
        "Pineapple": os.path.join(NEW_DATA_DIR, "pineapple_price.csv"),
        "Rice":      os.path.join(NEW_DATA_DIR, "rice_price.csv"),
        "Turmeric":  os.path.join(NEW_DATA_DIR, "turmeric_price_data.csv"),
        "Banana":    os.path.join(DATA_DIR, "banana_data.csv"),
        "Tapioca":   os.path.join(DATA_DIR, "tapioca_data.csv"),
    }

    results = []
    HOLDOUT_DAYS = 90   # ~3 months

    print(f"\n  {'Crop':<12} {'Samples':>8} {'Train':>7} {'Test':>6} {'MAE (₹)':>10} {'MAPE (%)':>10}  Status")
    print("  " + "-"*70)

    for crop, fpath in CROP_FILES.items():
        if not os.path.exists(fpath):
            print(f"  {crop:<12} {'—':>8} {'—':>7} {'—':>6} {'—':>10} {'—':>10}  ✗ file missing")
            continue
        try:
            df = pd.read_csv(fpath)
            df["ds"] = pd.to_datetime(df["t"], dayfirst=True, errors="coerce")
            df["y"]  = pd.to_numeric(df["p_modal"], errors="coerce")
            df = df[["ds", "y"]].dropna().sort_values("ds").drop_duplicates("ds")

            if len(df) < 10:
                print(f"  {crop:<12} {len(df):>8} {'—':>7} {'—':>6} {'—':>10} {'—':>10}  ✗ too few rows")
                continue

            cutoff    = df["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
            train_df  = df[df["ds"] <= cutoff]
            test_df   = df[df["ds"] >  cutoff]

            if len(train_df) < 5 or test_df.empty:
                print(f"  {crop:<12} {len(df):>8} {len(train_df):>7} {len(test_df):>6} {'—':>10} {'—':>10}  ✗ not enough split")
                continue

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, growth="flat")
            m.fit(train_df)

            future   = m.make_future_dataframe(periods=HOLDOUT_DAYS, freq="D")
            forecast = m.predict(future)
            forecast = forecast[["ds", "yhat"]].set_index("ds")

            test_df  = test_df.set_index("ds")
            joined   = test_df.join(forecast, how="inner")

            if joined.empty:
                print(f"  {crop:<12} {len(df):>8} {len(train_df):>7} {len(test_df):>6} {'—':>10} {'—':>10}  ✗ no overlap")
                continue

            mae  = (joined["y"] - joined["yhat"]).abs().mean()
            mape = ((joined["y"] - joined["yhat"]).abs() / joined["y"]).mean() * 100

            results.append({"crop": crop, "mae": mae, "mape": mape,
                            "n_total": len(df), "n_train": len(train_df), "n_test": len(joined)})

            status = "✓" if mape < 20 else ("△" if mape < 40 else "✗ high error")
            print(f"  {crop:<12} {len(df):>8} {len(train_df):>7} {len(joined):>6} {mae:>10.0f} {mape:>10.1f}  {status}")

        except Exception as e:
            print(f"  {crop:<12} {'—':>8} {'—':>7} {'—':>6} {'—':>10} {'—':>10}  ✗ {e}")

    if results:
        res_df = pd.DataFrame(results)
        print("\n  " + "-"*70)
        print(f"  {'AVERAGE':<12} {'':>8} {'':>7} {'':>6} {res_df['mae'].mean():>10.0f} {res_df['mape'].mean():>10.1f}")
        print(f"\n  Best  crop : {res_df.loc[res_df['mape'].idxmin(), 'crop']}  ({res_df['mape'].min():.1f}% MAPE)")
        print(f"  Worst crop : {res_df.loc[res_df['mape'].idxmax(), 'crop']}  ({res_df['mape'].max():.1f}% MAPE)")

        good  = (res_df["mape"] < 20).sum()
        ok    = ((res_df["mape"] >= 20) & (res_df["mape"] < 40)).sum()
        poor  = (res_df["mape"] >= 40).sum()
        print(f"\n  Accuracy bands:")
        print(f"    ✓ < 20% MAPE  (good)   : {good} crops")
        print(f"    △ 20–40% MAPE (ok)     : {ok} crops")
        print(f"    ✗ > 40% MAPE  (poor)   : {poor} crops")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│         AgriSetu — Model Accuracy Evaluation           │")
    print("└─────────────────────────────────────────────────────────┘")

    evaluate_suitability_model()
    evaluate_price_model()

    print("\n" + "="*60)
    print("  Evaluation complete.")
    print("="*60 + "\n")
