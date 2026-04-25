import os
import pandas as pd

# -------------------------------
# DOMAIN KNOWLEDGE BASE
# -------------------------------

# 1. Average Yields (in Quintals/Acre)
# Used to calculate concrete Revenue/ROI
# Source: General Agronomic Estimates for Kerala
# Average yields in Quintals/Acre (100 kg = 1 Quintal)
# Sources: Kerala Agriculture Dept., ICAR-CPCRI, KAU crop production guides
YIELD_ESTIMATES = {
    "Paddy":     25,    # 2,500 kg/acre — kharif + rabi average
    "Rice":      25,    # same as Paddy
    "Banana":    60,    # 6,000 kg/acre (single cycle harvest)
    "Tapioca":   50,    # 5,000 kg/acre (single cycle - 9 months)
    "Coconut":   45,    # ~4,500 nuts treated as quintals
    "Cashew":    8,     
    "Arecanut":  10,    
    "Pepper":    5,     
    "Coffee":    6,     
    "Ginger":    50,    
    "Turmeric":  40,    
    "Pineapple": 50,    # 5,000 kg/acre (single main harvest)
    "Papaya":    40,    # 4,000 kg/acre (first major harvest cycle)
    "Jackfruit": 60,    # 6,000 kg/acre (approx. 150 fruits × 40 kg average)
    "Rubber":    15,    # 1,500 kg/acre (dry rubber sheets)
    "Wheat":     15,    # Not common in Kerala — fallback value
}

# Dynamically update YIELD_ESTIMATES from kerala_monthly_estimates_2023_2025.csv if available
try:
    _csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "kerala_monthly_estimates_2023_2025.csv")
    if os.path.exists(_csv_path):
        _yield_df = pd.read_csv(_csv_path)
        _yield_df = _yield_df.dropna(subset=['Estimated_Productivity_kg_per_ha'])
        if not _yield_df.empty:
            # Average kg/ha across all months/years for each crop
            _avg_yield_kg_ha = _yield_df.groupby('Crop')['Estimated_Productivity_kg_per_ha'].mean()
            # Convert kg/ha to Quintals/Acre (1 ha = 2.47105 acres, 1 Quintal = 100 kg) -> divide by 247.105
            for _crop, _val in _avg_yield_kg_ha.items():
                _crop_title = str(_crop).title()
                _quintals_per_acre = round(_val / 247.105, 2)
                
                # Coconut data in CSV seems anomalous (e.g. 7.21). Keep standard for Coconut.
                if _crop_title == 'Coconut':
                    continue
                    
                if _crop_title == "Paddy":
                    YIELD_ESTIMATES["Rice"] = _quintals_per_acre
                YIELD_ESTIMATES[_crop_title] = _quintals_per_acre
except Exception as e:
    print(f"Warning: Could not load dynamic yield data: {e}")

# Base prices in INR/Quintal — fallback when Prophet prediction is unavailable or unreliable.
# Updated to match 4-year average from new AGMARK data (data/New/).
BASE_PRICES = {
    "Paddy":     2000,
    "Rice":      3800,
    "Banana":    2500,
    "Coconut":   3600,
    "Tapioca":   1800,
    "Cashew":    9400,
    "Arecanut":  35000,   # Updated: 4-yr avg from new data
    "Pepper":    42000,   # Updated: 4-yr avg from new data
    "Coffee":    12000,
    "Ginger":    10600,   # Updated: 4-yr avg from new data
    "Turmeric":  7100,    # Updated: 4-yr avg from new data
    "Pineapple": 4200,    # Added: 4-yr avg from new data
    "Papaya":    3500,    # Updated: 4-yr avg from new data
    "Jackfruit": 2500,    # Added: 4-yr avg from new data
    "Rubber":    15000,
    "Wheat":     2400,
}

# 2. Soil Preferences per crop for Kerala soil types
# 0 = Poor/Incompatible, 1 = Acceptable, 2 = Ideal
# Source: KAU Crop Production Guide & ICAR soil-crop suitability charts
SOIL_COMPATIBILITY = {
    # Clay soils               Loamy  Sandy  Red    Laterite
    "Paddy":     {"Clay": 2, "Loamy": 2, "Sandy": 0, "Red": 1, "Laterite": 0},  # needs water retention
    "Rice":      {"Clay": 2, "Loamy": 2, "Sandy": 0, "Red": 1, "Laterite": 0},
    "Coconut":   {"Clay": 1, "Loamy": 2, "Sandy": 2, "Red": 2, "Laterite": 2},  # very adaptable
    "Banana":    {"Clay": 1, "Loamy": 2, "Sandy": 1, "Red": 1, "Laterite": 1},  # needs drainage
    "Tapioca":   {"Clay": 0, "Loamy": 2, "Sandy": 2, "Red": 2, "Laterite": 2},  # roots need loose soil
    "Cashew":    {"Clay": 0, "Loamy": 1, "Sandy": 2, "Red": 2, "Laterite": 2},  # drought-tolerant, laterite native
    "Arecanut":  {"Clay": 1, "Loamy": 2, "Sandy": 0, "Red": 2, "Laterite": 1},  # well-drained loamy/red
    "Pepper":    {"Clay": 1, "Loamy": 2, "Sandy": 0, "Red": 2, "Laterite": 2},  # forest loam & laterite
    "Coffee":    {"Clay": 0, "Loamy": 2, "Sandy": 0, "Red": 2, "Laterite": 2},  # Wayanad red/laterite best
    "Ginger":    {"Clay": 0, "Loamy": 2, "Sandy": 1, "Red": 2, "Laterite": 1},  # well-drained red loam
    "Turmeric":  {"Clay": 0, "Loamy": 2, "Sandy": 1, "Red": 2, "Laterite": 1},  # similar to ginger
    "Pineapple": {"Clay": 0, "Loamy": 2, "Sandy": 2, "Red": 2, "Laterite": 2},  # well-drained, acidic OK
    "Papaya":    {"Clay": 0, "Loamy": 2, "Sandy": 2, "Red": 1, "Laterite": 1},  # needs good drainage
    "Jackfruit": {"Clay": 1, "Loamy": 2, "Sandy": 1, "Red": 2, "Laterite": 2},  # very hardy, wide tolerance
    "Rubber":    {"Clay": 1, "Loamy": 2, "Sandy": 0, "Red": 2, "Laterite": 2},
}

# 3. Companion / Intercropping suggestions
# Based on Kerala intercropping practices (KAU extension handbook)
COMPANION_MAP = {
    "Coconut":   ["Pineapple", "Ginger", "Turmeric", "Pepper", "Banana"],
    "Arecanut":  ["Pepper", "Banana", "Cocoa", "Vanilla"],
    "Rubber":    ["Pineapple", "Banana"],
    "Coffee":    ["Pepper", "Cardamom", "Banana"],
    "Banana":    ["Ginger", "Turmeric", "Elephant Foot Yam"],
    "Tapioca":   ["Banana", "Groundnut"],
    "Pepper":    ["Arecanut", "Coconut"],          # pepper grown on live standards
    "Jackfruit": ["Pepper", "Ginger", "Pineapple"],
    "Paddy":     ["Azolla"],                       # bio-fertiliser intercrop
    "Rice":      ["Azolla"],
    "Pineapple": ["Coconut", "Cashew"],
    "Ginger":    ["Coconut", "Areca", "Banana"],
    "Turmeric":  ["Coconut", "Areca"],
}

def parse_duration_to_days(duration_str):
    """
    Parses strings like '120 days', '4 months', '1 year' into integer days.
    """
    if isinstance(duration_str, (int, float)):
        return int(duration_str)
        
    d_str = str(duration_str).lower().strip()
    
    # Identify unit first to prevent losing it if we split a range (e.g., '5-7 years')
    unit = ""
    if 'month' in d_str: unit = 'month'
    elif 'year' in d_str: unit = 'year'
    elif 'day' in d_str: unit = 'day'

    if '-' in d_str:
        d_str = d_str.split('-')[0].strip()
        # Restore unit if it was lost in split
        if unit and unit not in d_str:
            d_str += f" {unit}"
    
    if 'month' in d_str:
        try:
            val = float(d_str.split()[0])
            return int(val * 30)
        except:
            return 90 # Default fallback
            
    if 'year' in d_str:
        try:
            val = float(d_str.split()[0])
            return int(val * 365)
        except:
            return 365
            
    if 'day' in d_str:
        try:
            val = float(d_str.split()[0])
            return int(val)
        except:
            return 90
            
    try:
        return int(d_str)
    except:
        return 90
