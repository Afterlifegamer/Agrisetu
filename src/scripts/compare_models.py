import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR  = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
NEW_DATA_DIR = os.path.join(DATA_DIR, "New")

sys.path.insert(0, SRC_DIR)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import data_utils
from suitability_model import CropSuitabilityModel

def get_suitability_data():
    sm = CropSuitabilityModel()
    crop_data = sm.crop_data
    yield_file = os.path.join(DATA_DIR, "kerala_monthly_estimates_2023_2025.csv")
    yield_df = pd.read_csv(yield_file)

    from dateutil.relativedelta import relativedelta
    import datetime

    training_rows = []
    # simplified recreation so it matches evaluate_models.py
    for _, row in crop_data.iterrows():
        crop_name = str(row["crop_name"]).strip()
        duration_days   = data_utils.parse_duration_to_days(row["duration_to_harvest"])
        duration_months = max(1, round(duration_days / 30.0))

        crop_yields = yield_df[yield_df["Crop"].str.lower() == crop_name.lower()]
        if crop_yields.empty: continue

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

            phase_stats = sm._get_phase_stats(harvest_month, duration_months, direction='backward')
            
            row_data = {
                "water_dependency_num":  row["water_dependency_num"],
                "time_effort_num":       row["time_effort_num"],
                "price_at_planting":     price_at_plant,
                "label":                 label,
            }
            row_data.update(phase_stats)
            training_rows.append(row_data)

    df_train = pd.DataFrame(training_rows)
    X = df_train[[
        "rain_sowing", "temp_sowing", "hum_sowing",
        "rain_growth", "temp_growth", "hum_growth",
        "rain_harvest", "temp_harvest", "hum_harvest",
        "water_dependency_num", "time_effort_num", "price_at_planting"
    ]]
    y = df_train["label"]
    return X, y

def get_price_data(crop_name):
    fpath = os.path.join(NEW_DATA_DIR, f"{crop_name.lower()}_price.csv")
    if not os.path.exists(fpath):
        fpath = os.path.join(NEW_DATA_DIR, f"{crop_name.lower()}_price_data.csv")
        
    df = pd.read_csv(fpath)
    df["ds"] = pd.to_datetime(df["t"], dayfirst=True, errors="coerce")
    df["y"]  = pd.to_numeric(df["p_modal"], errors="coerce")
    df = df[["ds", "y"]].dropna().sort_values("ds").drop_duplicates("ds")
    return df

def main():
    print("Gathering data...")
    X_suit, y_suit = get_suitability_data()

    print("Evaluating Suitability Models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    suit_models = [
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42))
    ]
    
    suit_results = {}
    for name, model in suit_models:
        scores = cross_val_score(model, X_suit, y_suit, cv=cv, scoring="accuracy")
        suit_results[name] = scores.mean() * 100
        print(f"  {name}: {suit_results[name]:.2f}%")

    print("\nEvaluating Price Models...")
    # Evaluate across a few distinct crops to average
    crops_to_test = ["Arecanut", "Cashew", "Pineapple"]
    
    prophet_mapes = []
    xgb_mapes = []
    linear_mapes = []
    
    for crop in crops_to_test:
        df = get_price_data(crop)
        HOLDOUT_DAYS = 90
        cutoff = df["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        train_df = df[df["ds"] <= cutoff]
        test_df  = df[df["ds"] >  cutoff]
        
        # PROPHET
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, growth="flat")
        m_prophet.fit(train_df)
        future = m_prophet.make_future_dataframe(periods=len(test_df), freq="D")
        forecast = m_prophet.predict(future)
        joined = test_df.set_index("ds").join(forecast.set_index("ds"), how="inner")
        prophet_mapes.append((joined["y"] - joined["yhat"]).abs() / joined["y"].replace(0, 1) * 100)
        
        # XGBRegressor
        # create time-based features
        def make_features(d):
            return pd.DataFrame({
                "dayofyear": d["ds"].dt.dayofyear,
                "month": d["ds"].dt.month,
                "year": d["ds"].dt.year
            }, index=d.index)
        
        X_train_ts = make_features(train_df)
        y_train_ts = train_df["y"]
        X_test_ts = make_features(test_df)
        y_test_ts = test_df["y"]
        
        m_xgb = XGBRegressor(n_estimators=100, random_state=42)
        m_xgb.fit(X_train_ts, y_train_ts)
        xgb_preds = m_xgb.predict(X_test_ts)
        xgb_mapes.append((y_test_ts - xgb_preds).abs() / y_test_ts.replace(0, 1) * 100)
        
        # Logistic Regression / Linear baseline
        m_lin = LinearRegression()
        m_lin.fit(X_train_ts, y_train_ts)
        lin_preds = m_lin.predict(X_test_ts)
        linear_mapes.append((y_test_ts - lin_preds).abs() / y_test_ts.replace(0, 1) * 100)
        
    price_results = {
        "Prophet": np.mean(pd.concat(prophet_mapes)),
        "XGBoost\n(Time Series)": np.mean(pd.concat(xgb_mapes)),
        "Linear\nRegression": np.mean(pd.concat(linear_mapes))
    }
    
    for name, mape in price_results.items():
        print(f"  {name.replace(chr(10), ' ')}: {mape:.2f}% MAPE")
        
    print("\nGenerating comparative graph...")
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Classification Accuracy
    names_suit = list(suit_results.keys())
    accs_suit = list(suit_results.values())
    colors_suit = ['#2ca02c' if name == 'XGBoost' else '#1f77b4' for name in names_suit]
    
    bars1 = ax1.bar(names_suit, accs_suit, color=colors_suit)
    ax1.set_title("Crop Suitability Classification\n(Higher is Better)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cross-Validation Accuracy (%)", fontsize=12)
    ax1.set_ylim([max(0, min(accs_suit)-10), min(100, max(accs_suit)+10)])
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Plot Price Forecast MAPE
    names_price = list(price_results.keys())
    errs_price = list(price_results.values())
    colors_price = ['#d62728' if name == 'Prophet' else '#1f77b4' for name in names_price]

    bars2 = ax2.bar(names_price, errs_price, color=colors_price)
    ax2.set_title("Crop Price Prediction Error\n(Lower is Better)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Mean Absolute Percentage Error (MAPE %)", fontsize=12)
    ax2.set_ylim([0, max(errs_price) + 5])

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_path = r"C:\Users\abhir\.gemini\antigravity\brain\480d6002-c851-464f-892f-5f83ece23eb4\scratch\model_comparison.png"
    
    # ensure scratch exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Graph generation complete! Saved to {out_path}")


if __name__ == "__main__":
    main()
