import requests
from datetime import datetime
import json

# Approximate coordinates for Kerala Districts
DISTRICT_COORDS = {
    "Kottayam": {"lat": 9.59, "lon": 76.52},
    "Alappuzha": {"lat": 9.49, "lon": 76.33},
    "Idukki": {"lat": 9.85, "lon": 76.97},
    "Thiruvananthapuram": {"lat": 8.52, "lon": 76.93},
    "Ernakulam": {"lat": 9.98, "lon": 76.28},
    "Thrissur": {"lat": 10.52, "lon": 76.21},
    "Palakkad": {"lat": 10.78, "lon": 76.65},
    "Malappuram": {"lat": 11.05, "lon": 76.07},
    "Kozhikode": {"lat": 11.25, "lon": 75.78},
    "Wayanad": {"lat": 11.68, "lon": 76.13},
    "Kannur": {"lat": 11.87, "lon": 75.37},
    "Kasaragod": {"lat": 12.51, "lon": 74.98},
    "Kollam": {"lat": 8.89, "lon": 76.61},
    "Pathanamthitta": {"lat": 9.26, "lon": 76.78}
}

def get_realtime_weather_forecast(district_name):
    """
    Fetches the 7-day forecast for a district using OpenMeteo (Free, No Key).
    Returns: Average Rainfall (mm), Average Temp (C)
    """
    print(f"☁️  Fetching real-time weather for {district_name}...")
    
    # 1. Get Coords
    district_clean = district_name.title().strip()
    coords = DISTRICT_COORDS.get(district_clean)
    
    if not coords:
        print(f"⚠️  District '{district_clean}' not found in coordinates map. Using default/fallback.")
        coords = DISTRICT_COORDS["Kottayam"] # Fallback

    # 2. Call API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "daily": ["temperature_2m_max", "precipitation_sum"],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if "daily" in data:
            # Calculate Averages for the next 7 days
            temps = data["daily"]["temperature_2m_max"]
            rains = data["daily"]["precipitation_sum"]
            
            avg_temp = sum(temps) / len(temps)
            avg_rain = sum(rains) / len(rains)
            
            # Simple "Season" classification based on realtime rain
            if avg_rain > 10:
                current_season = "Heavy Rain"
            elif avg_rain > 2:
                current_season = "Moderate Rain"
            else:
                current_season = "Dry/Sunny"
                
            print(f"   -> Forecast (7 days): {current_season} | Temp: {avg_temp:.1f}°C | Rain: {avg_rain:.1f}mm/day")
            return {
                "avg_temp": avg_temp,
                "avg_rain": avg_rain,
                "season": current_season,
                "valid": True
            }
            
    except Exception as e:
        print(f"❌ Weather API Error: {e}")
    
    return {"valid": False, "avg_rain": 15.0} # Fallback to moderate

def get_historical_weather_forecast(district_name):
    """
    Returns average historical weather for the current month.
    """
    import datetime
    current_month = datetime.datetime.now().month
    
    if historical_scanner.df is None:
        return {"valid": False, "avg_rain": 0, "avg_temp": 28, "season": "N/A"}
        
    district_data = historical_scanner.df[historical_scanner.df['district'].str.contains(district_name, case=False, na=False)]
    if district_data.empty:
        district_data = historical_scanner.df
        
    m_data = district_data[district_data['month'] == current_month]
    if m_data.empty:
        return {"valid": False, "avg_rain": 0, "avg_temp": 28, "season": "N/A"}
        
    avg_temp = m_data['temp_2m_C'].mean()
    avg_rain = m_data['rain_mm'].mean()
    
    if avg_rain > 10:
        current_season = "Monsoon"
    elif avg_rain > 2:
        current_season = "Post-Monsoon"
    else:
        current_season = "Summer/Dry"
        
    return {
        "avg_temp": avg_temp,
        "avg_rain": avg_rain,
        "season": current_season,
        "valid": True
    }

class LongTermWeatherScanner:
    def __init__(self, history_file=None):
        if history_file is None:
             import os
             BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             history_file = os.path.join(BASE_DIR, 'data', "kerala_weather_2023_2024_full.csv")
             
        self.history_file = history_file
        self.df = None
        self._load_data()
        
    def _load_data(self):
        try:
            import pandas as pd
            self.df = pd.read_csv(self.history_file)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['month'] = self.df['date'].dt.month
        except Exception as e:
            print(f"⚠️  Could not load historical weather: {e}")
            self.df = None

    def analyze_risk(self, district, start_month, duration_months):
        """
        Scans future months for extreme risks based on history.
        """
        if self.df is None:
            return 0.0, []
        
        district_data = self.df[self.df['district'].str.contains(district, case=False, na=False)]
        if district_data.empty:
            # Fallback to general Kerala average if district not found
            district_data = self.df
            
        risks = []
        total_risk_penalty = 0.0
        
        # Monthly averages reference (approx from data)
        # Check specific months in duration
        current_m = start_month
        
        for i in range(duration_months):
            # 1-12 cycle
            check_month = ((current_m + i - 1) % 12) + 1
            
            # Get historical avg for this month
            m_data = district_data[district_data['month'] == check_month]
            if m_data.empty:
                continue
                
            avg_rain = m_data['rain_mm'].mean() * 30 # Monthly total approx
            
            # Risk Logic
            if avg_rain > 500: # Heavy Monsoon (June/July level)
                risks.append(f"Month {i+1} (M{check_month}): High Flood Risk ({int(avg_rain)}mm)")
                total_risk_penalty += 0.3 # Heavy penalty
            elif avg_rain < 10 and check_month in [1, 2, 3]: # Dry Season (Jan-Mar)
                risks.append(f"Month {i+1} (M{check_month}): Drought Risk ({int(avg_rain)}mm)")
                total_risk_penalty += 0.2
                
        return min(1.0, total_risk_penalty), risks

# Singleton instance
historical_scanner = LongTermWeatherScanner()

if __name__ == "__main__":
    get_realtime_weather_forecast("Kottayam")
    # Test Long Term
    print("\n--- Long Term Scan Test (Kottayam, Start=Oct, Dur=6m) ---")
    risk, msgs = historical_scanner.analyze_risk("Kottayam", 10, 6)
    print(f"Risk Score: {risk}")
    print("Warnings:", msgs)
