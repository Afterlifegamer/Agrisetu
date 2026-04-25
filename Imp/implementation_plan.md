# Phase-Based Growth Modeling (XGBoost Implementation)

This blueprint explains how to implement "Monthly/Stage-Aware" biological logic without leaving the XGBoost framework. 

## 1. Feature Engineering: The 3-Phase Logic
Instead of a single "Average Rainfall" column, we split every crop's duration into three biologically significant buckets:

```python
def get_phase_stats(self, start_month, duration_months):
    # Divide duration into 3 chunks: 
    # Phase 1: Establishment (first 25%)
    # Phase 2: Growth (middle 50%)
    # Phase 3: Ripening/Harvest (last 25%)
    
    p1_end = max(1, round(duration_months * 0.25))
    p2_end = max(p1_end + 1, round(duration_months * 0.75))
    
    phases = {
        "Sowing": (1, p1_end),
        "Growth": (p1_end + 1, p2_end),
        "Harvest": (p2_end + 1, duration_months)
    }
    
    features = {}
    for name, (start, end) in phases.items():
        # Calculate avg rain/temp/hum for THIS specific slice of time
        # ... logic to slice self.weather_data ...
        features[f"rain_{name}"] = avg_rain
        features[f"temp_{name}"] = avg_temp
        features[f"hum_{name}"] = avg_hum
        
    return features
```

## 2. Updated XGBoost Feature Vector
Previously, we had 4 features. Now, we use **12 features** to give the AI "Temporal Vision":

| Feature Category | Columns |
| :--- | :--- |
| **Sowing Phase** | `rain_sowing`, `temp_sowing`, `hum_sowing` |
| **Growth Phase** | `rain_growth`, `temp_growth`, `hum_growth` |
| **Harvest Phase** | `rain_harvest`, `temp_harvest`, `hum_harvest` |
| **Fixed Attributes** | `water_dependency`, `time_effort`, `starting_price` |

## 3. Training & Prediction
- **Training:** During the [train()](file:///c:/Users/abhir/Desktop/Final%20Project/src/suitability_model.py#108-182) loop, when we find a historical harvest (e.g., Paddy in Oct 2023), we look back 4 months and calculate the 3-phase climate stats to feed into XGBoost.
- **Inference:** When a farmer asks for a recommendation *now*, we use the **Weather Forecast** for the next N months, split into these 3 phases, and ask XGBoost for a score.

## 📈 Why this works for XGBoost:
XGBoost only accepts a fixed number of inputs. By consolidating any crop duration (3 months or 12 months) into these **three consistent phases**, we keep the column count stable (12 columns) while giving the AI a massive biological upgrade.
