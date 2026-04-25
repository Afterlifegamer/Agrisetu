import pandas as pd
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore')
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

import importlib.util
import data_utils
import weather_service

import suitability_model
import crop_reco

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")



def _load_yield_profiles():
    yield_path = os.path.join(MODELS_DIR, "crop_yield_profiles.json")
    if os.path.exists(yield_path):
        try:
            with open(yield_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: could not load yield profiles ({e}). Using defaults.")
    return {}




def hybrid_recommendation(district, max_budget, max_duration_months, soil_type="Loamy"):
    print(f"\n Generating Hybrid Recommendations for: {district}")
    print(f" Budget: ₹{max_budget}/acre |  Max Duration: {max_duration_months} months |  Soil: {soil_type}")
    
    yield_profiles = _load_yield_profiles()
    
    print("-" * 60)

    weather_ctx = weather_service.get_realtime_weather_forecast(district)
    if weather_ctx["valid"]:
        print(f"  Live Forecast: {weather_ctx['season']} (Temp: {weather_ctx['avg_temp']:.1f}°C, Rain: {weather_ctx['avg_rain']:.1f}mm)")
    
    print("... Analyzing soil and weather suitability ...")
    suitability_db = suitability_model.CropSuitabilityModel()
    import datetime
    current_month = datetime.datetime.now().month
    suitability_df = suitability_db.predict_suitability(district, current_month=current_month)
    
    suitability_df['days_to_harvest'] = suitability_df['duration_to_harvest'].apply(data_utils.parse_duration_to_days)
    suitability_df['approx_months'] = suitability_df['days_to_harvest'] / 30
    
    suitability_df['total_cost'] = suitability_df['initial_cost_inr_per_acre'] + suitability_df['maintenance_cost_inr_per_acre']
    filtered_df = suitability_df[
        (suitability_df['total_cost'] <= max_budget) & 
        (suitability_df['approx_months'] <= max_duration_months)
    ].copy()
    
    if filtered_df.empty:
        print(" No crops matched your budget and duration constraints.")
        return
    
    duration_map = dict(zip(filtered_df.crop_name, filtered_df.days_to_harvest))
    
    print("... Predicting market prices & Volatility ...")
    price_df = crop_reco.recommend_crops_by_location(district, duration_map)
    
    if not price_df.empty:
        if 'crop' in price_df.columns:
            price_df = price_df.rename(columns={'crop': 'crop_name'})
            
    final_df = pd.merge(filtered_df, price_df, on="crop_name", how="left")
    
    def get_fallback_price(row):
        if pd.notna(row['predicted_price']) and row['predicted_price'] > 0:
            return row['predicted_price']
        
        return data_utils.BASE_PRICES.get(row['crop_name'], 
               data_utils.BASE_PRICES.get(row['crop_name'].capitalize(), 0))

    final_df['predicted_price'] = final_df.apply(get_fallback_price, axis=1)
    final_df['volatility_index'] = final_df['volatility_index'].fillna(0.2) 
    
    roi_scores = []
    soil_scores = []
    risk_notes = []
    expected_yields = []
    profits = []
    revenues = []
    harvest_costs = []
    
    for _, row in final_df.iterrows():
        crop = row['crop_name']
        initial_cost = row['initial_cost_inr_per_acre']
        maintenance_cost = row['maintenance_cost_inr_per_acre']
        duration_m = row['approx_months'] if row['approx_months'] > 0 else 4
        
        predicted_label = str(int(row.get('predicted_label', 1)))
        
        # Evaluate Physical Soil Constraint FIRST
        soil_prefs = data_utils.SOIL_COMPATIBILITY.get(crop, {})
        soil_compatibility = soil_prefs.get(soil_type, 1) 
        
        if soil_compatibility == 0:
            s_factor = 0.2  # Severe penalty for crop-soil mismatch (e.g., Tapioca in Clay)
            risk_notes.append(f" ⚠️ Poor Soil ({soil_type})")
        elif soil_compatibility == 2:
            s_factor = 1.25 # Boost for Ideal soil
            risk_notes.append(" ✅ Ideal Soil")
        else:
            s_factor = 1.0
            risk_notes.append("")
            
        soil_scores.append(s_factor)
        
        profile = yield_profiles.get(crop)
        if profile and predicted_label in profile:
            raw_yield = float(profile[predicted_label])
        else:
            raw_yield = data_utils.YIELD_ESTIMATES.get(crop, 10) 
            
        # STRICT BIOLOGY: Penalize or boost the literal crop size based on soil match
        est_yield = raw_yield * s_factor
            
        expected_yields.append(est_yield)
        
        revenue_per_harvest = est_yield * row['predicted_price']
        
        # Add post-harvest and labor margins (approx 30% of revenue usually goes to harvesting/transport/labor)
        harvest_and_labor_cost = revenue_per_harvest * 0.30
        
        total_cost = initial_cost + maintenance_cost + harvest_and_labor_cost
        profit = revenue_per_harvest - total_cost
        
        roi = (profit / total_cost) * 100 if total_cost > 0 else 0
        roi_scores.append(roi)
        profits.append(profit)
        revenues.append(revenue_per_harvest)
        harvest_costs.append(harvest_and_labor_cost)

    final_df['est_roi'] = roi_scores
    final_df['soil_factor'] = soil_scores
    final_df['risk_note'] = risk_notes
    final_df['dynamic_expected_yield'] = expected_yields
    final_df['profit_cycle'] = profits
    final_df['revenue'] = revenues
    final_df['harvest_cost'] = harvest_costs
    
    print("... Calculating Risk-Adjusted ROI ...")
    
    # 1. Start with the explicit profit margin generated via biological yield and unit price
    base_score = final_df['est_roi']
    
    # 2. Discount the final score to account for historic market price volatility
    volatility_penalty = 1 - (final_df['volatility_index'].clip(0, 0.3))
    
    # The raw financial expectation (Yield was already penalized by Soil earlier)
    raw_hybrid_score = base_score * volatility_penalty
    
    # Scale from 0.0 to 1.0 so the frontend progress bars (which multiply by 100) render perfectly.
    max_score = raw_hybrid_score.max()
    final_df['hybrid_score'] = raw_hybrid_score.apply(lambda x: max(0.0, x / max_score) if max_score > 0 else 0.0)
    
    # Temporarily excluded crops (remove from list to re-enable)
    EXCLUDED_CROPS = ["Tapioca"]
    final_df = final_df[~final_df['crop_name'].isin(EXCLUDED_CROPS)]
    
    final_df = final_df.sort_values(by='hybrid_score', ascending=False)
    
    print("\nTop Recommended Crops:")
    print("=" * 110)
    
    # Exposing the dynamic expected yield so the user actually sees the effect!
    display_cols = ['crop_name', 'hybrid_score', 'est_roi', 'dynamic_expected_yield', 'predicted_price']
    
    header_map = {
        'crop_name': 'Crop',
        'hybrid_score': 'Score',
        'est_roi': 'Est. ROI (%)',
        'dynamic_expected_yield': 'Yield (Q/Acre)',
        'predicted_price': 'Pred. Price'
    }
    
    view_df = final_df[display_cols].rename(columns=header_map)
    
    print(view_df.to_string(index=False, formatters={
        'Score': '{:.2f}'.format,
        'Est. ROI (%)': '{:.0f}%'.format,
        'Yield (Q/Acre)': '{:.1f}'.format,
        'Pred. Price': '₹{:.0f}'.format
    }))
    print("=" * 110)
    
    best = final_df.iloc[0]
    print(f"\n🌟 WINNER: {best['crop_name']}")
    print(f"   💰 Expected Return: {best['est_roi']:.0f}% per acre")
    
    companions = data_utils.COMPANION_MAP.get(best['crop_name'])
    if companions:
        print(f"   💡 Pro Tip: Intercrop with {', '.join(companions)} to maximize land use!")
        
    return final_df

if __name__ == "__main__":
    print("\n---  Hybrid Crop Recommender System (v2.0)  ---")
    
    d_input = input("Enter District Name (Default: Thiruvananthapuram): ").strip() or "Thiruvananthapuram"
    
    try:
        b_str = input("Enter Max Budget in ₹ (Default: 50000): ").strip()
        budget_input = float(b_str) if b_str else 50000
    except ValueError:
        budget_input = 50000

    try:
        t_str = input("Enter Max Duration in Months (Default: 6): ").strip()
        duration_input = float(t_str) if t_str else 6
    except ValueError:
        duration_input = 6
        
    print("\nSoil Types: Clay, Sandy, Loamy, Red, Laterite")
    s_input = input("Enter Soil Type (Default: Loamy): ").strip().capitalize()
    if s_input not in ["Clay", "Sandy", "Loamy", "Red", "Laterite"]:
        s_input = "Loamy"
    
    hybrid_recommendation(d_input, budget_input, duration_input, s_input)
