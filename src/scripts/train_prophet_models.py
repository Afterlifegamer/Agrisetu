import pandas as pd
from prophet import Prophet
import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

NEW_DATA_DIR = os.path.join(DATA_DIR, 'New')


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

MIN_DAYS_FOR_YEARLY_SEASONALITY = 365   # need ≥ 1 year to learn annual patterns

def train_and_save_all_models():
    saved_models = {}
    needs_more_data = []

    print("Starting training for all crops...")

    for crop_name, file_path in CROP_FILES.items():
        print(f"   Training model for: {crop_name}...")

        if not os.path.exists(file_path):
            print(f"   WARNING: Data file not found: {file_path}")
            needs_more_data.append((crop_name, "FILE MISSING", 0))
            continue

        try:
            df = pd.read_csv(file_path)

            # Use state-level data for a robust general model
            # Group by date and take average price across all districts
            df['ds'] = pd.to_datetime(df['t'], dayfirst=True, errors='coerce')
            df['y'] = pd.to_numeric(df['p_modal'], errors='coerce')
            df = df[['ds', 'y']].dropna().sort_values('ds')

            # Aggregate if multiple entries for same date (e.g. multiple districts)
            daily_df = df.groupby('ds')['y'].mean().reset_index()

            if len(daily_df) < 5:
                print(f"   WARNING: Insufficient data points for {crop_name}. Skipping.")
                needs_more_data.append((crop_name, "< 5 daily points", 0))
                continue

            days_span = (daily_df['ds'].max() - daily_df['ds'].min()).days

            # Only enable yearly seasonality when we have at least 1 year of data.
            # With less data, Fourier terms extrapolate wildly months into the future.
            use_yearly = days_span >= MIN_DAYS_FOR_YEARLY_SEASONALITY
            if not use_yearly:
                print(f"   INFO: {crop_name} has only {days_span} days of data "
                      f"(< {MIN_DAYS_FOR_YEARLY_SEASONALITY}). "
                      f"Disabling yearly seasonality to avoid bad extrapolation.")
                needs_more_data.append((crop_name, f"only {days_span} days", days_span))

            # Train Prophet Model
            m = Prophet(
                yearly_seasonality=use_yearly,
                weekly_seasonality=False,
                daily_seasonality=False,
                growth="flat"
            )
            m.fit(daily_df)

            saved_models[crop_name] = m
            print(f"   OK: {crop_name} trained ({len(daily_df)} pts, {days_span} days, yearly_season={use_yearly})")

        except Exception as e:
            print(f"   ERROR training {crop_name}: {e}")

   
    output_path = os.path.join(MODELS_DIR, "prophet_models.joblib")
    joblib.dump(saved_models, output_path)
    print(f"\nAll models saved to: {output_path}")

    # ── Data quality report ──────────────────────────────────────────────────
    if needs_more_data:
        print("\n" + "="*60)
        print("  DATA QUALITY REPORT — Crops needing more historical data")
        print("="*60)
        print("  The following crops have < 1 year of price data.")
        print("  Their Prophet models use no yearly seasonality, which means")
        print("  price predictions are less accurate (flat trend only).")
        print("  To improve accuracy, collect 2+ years of AGMARK/eNAM data.\n")
        for c, reason, days in needs_more_data:
            needed = max(0, MIN_DAYS_FOR_YEARLY_SEASONALITY - days)
            print(f"  - {c:<12} ({reason}, need ~{needed} more days of data)")
        print("="*60)
    else:
        print("\nAll crops have sufficient data for yearly seasonality.")


if __name__ == "__main__":
    train_and_save_all_models()

