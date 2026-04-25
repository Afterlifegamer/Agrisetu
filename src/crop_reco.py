import pandas as pd
from prophet import Prophet
import os
import joblib
from datetime import datetime, timedelta
import data_utils


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
NEW_DATA_DIR = os.path.join(DATA_DIR, 'New')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Specific mapping of crop display names to actual CSV filenames.
# New 4-year data (data/New/) used where available; original files as fallback.
CROP_FILES = {
    # ── New 4-year data ────────────────────────────────────────────────────────
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
    # ── Original data (33-day window — no new data yet) ───────────────────────
    "Banana":    os.path.join(DATA_DIR, "banana_data.csv"),
    "Tapioca":   os.path.join(DATA_DIR, "tapioca_data.csv"),
}

PREDICTION_DAYS = 60   # 2 months

# LOAD SAVED MODELS (Unified Cache)
SAVED_MODELS = {}
MODEL_PATH = os.path.join(MODELS_DIR, "prophet_models.joblib")
if os.path.exists(MODEL_PATH):
    # print(f" Loaded {MODEL_PATH} for fast prediction.")
    SAVED_MODELS = joblib.load(MODEL_PATH)
else:
    print("  No saved price models found. Using slower live training.")



def recommend_crops_by_location(district, crop_durations=None):
    recommendations = []
    
    today = pd.to_datetime(datetime.today().date())
    
  
    default_target = today + timedelta(days=PREDICTION_DAYS)

    for crop in CROP_FILES.keys():
        if crop_durations and crop in crop_durations:
            days = crop_durations[crop]

            target_date = today + timedelta(days=days)
        elif crop_durations and crop.capitalize() in crop_durations:
             days = crop_durations[crop.capitalize()]
             target_date = today + timedelta(days=days)
        else:
            target_date = default_target
            
        file_name = CROP_FILES[crop] 
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name)
                df_loc = df[df['district_name'] == district].copy()
                if not df_loc.empty:
                    df_loc['ds'] = pd.to_datetime(df_loc['t'], dayfirst=True, errors='coerce')
                    df_loc['y'] = df_loc['p_modal']
                    df_loc = df_loc[['ds', 'y']].dropna().sort_values('ds')
                else:
                    df_loc = pd.DataFrame(columns=['ds', 'y']) 
            except:
                df_loc = pd.DataFrame(columns=['ds', 'y'])
        else:
             df_loc = pd.DataFrame(columns=['ds', 'y'])

       
        model = None
        
       
        if crop in SAVED_MODELS:
            model = SAVED_MODELS[crop]
        elif crop.capitalize() in SAVED_MODELS:
            model = SAVED_MODELS[crop.capitalize()]
            
        
        if model is None:
            
            if not df_loc.empty and len(df_loc) >= 5:
                try:
                    m = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        growth="flat"
                    )
                    m.fit(df_loc)
                    model = m
                except:
                    model = None

       
        future = pd.DataFrame({'ds': [target_date]})
        
        predicted_price = 0
        volatility_index = 0.2 

        if model:
            try:
                forecast = model.predict(future)
                predicted_price = forecast.iloc[0]['yhat']
                
               
                if not df_loc.empty:
                    volatility = df_loc['y'].std()
                    mean_price = df_loc['y'].mean()
                    volatility_index = (volatility / mean_price) if mean_price > 0 else 0
            except:
                predicted_price = 0

        # FALLBACK 1: State Average
        if predicted_price <= 0:
            # We use the CROP_FILES mapping already defined at the top of the loop
            file_name = CROP_FILES.get(crop, CROP_FILES.get(crop.capitalize()))
            if file_name and os.path.exists(file_name):
                 try:
                     df_fallback = pd.read_csv(file_name)
                     if not df_fallback.empty and 'p_modal' in df_fallback.columns:
                         predicted_price = df_fallback['p_modal'].mean()
                         # Volatility for state average
                         volatility_index = (df_fallback['p_modal'].std() / predicted_price) if predicted_price > 0 else 0.2
                 except:
                     pass

        # FALLBACK 2: Hardcoded Base Price
        if predicted_price <= 0:
            # keys in BASE_PRICES might be "Rice", "Coconut" etc.
            predicted_price = data_utils.BASE_PRICES.get(crop, data_utils.BASE_PRICES.get(crop.capitalize(), 2000))
            volatility_index = 0.1 # Low risk assumption for static price

        recommendations.append({
            "crop": crop,
            "predicted_price": round(predicted_price, 2),
            "volatility_index": round(volatility_index, 3), 
            "harvest_date": target_date.strftime("%Y-%m-%d")
        })

    if not recommendations:
        return pd.DataFrame(columns=["crop", "predicted_price", "volatility_index", "harvest_date"])

    # Sort crops by highest predicted price
    recommendations = sorted(
        recommendations,
        key=lambda x: x['predicted_price'],
        reverse=True
    )

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    district_input = "Kottayam"
    print(f"Predicting prices for {district_input}...")
    result = recommend_crops_by_location(district_input)
    print(result)
