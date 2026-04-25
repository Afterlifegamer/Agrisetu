# Phase 6: Biological Logic Expansion (Scientific Upgrade)

This plan details the two-pronged upgrade to the AgriSetu biological intelligence layer: **Growth-Stage Sensitive Features** and **Dynamic Yield Profiling**.

## 1. Growth-Stage Sensitive Features (XGBoost)
Instead of a single "Average" climate value, we split every crop's lifecycle into three biologically significant buckets to allow XGBoost to learn stage-specific needs:

- **Sowing Phase (First 25% of duration):** Learns the optimal temperature and moisture for germinating roots.
- **Mid-Growth Phase (Middle 50%):** Learns the sustained requirements for building biomass/leaves.
- **Harvest Phase (Final 25%):** Learns which crops need "dry-downs" or specific humidity levels for ripening.

### 📊 New Feature Vector (12 Features):
| Phase | Features |
| :--- | :--- |
| **Establishment** | `rain_sowing`, `temp_sowing`, `hum_sowing` |
| **Growth** | `rain_growth`, `temp_growth`, `hum_growth` |
| **Maturation** | `rain_harvest`, `temp_harvest`, `hum_harvest` |
| **Attributes** | `water_dep`, `time_effort`, `starting_price` |

---

## 2. Dynamic "Three-Point" Yield Profiling
We replace the single `yield_estimate` average with a **3-Point Profile** based on historical Kerala production data:

- **Label 2 (Highly Suitable):** Maps to the **90th Percentile** (Best Case Scenario).
- **Label 1 (Moderately Suitable):** Maps to the **50th Percentile / Median** (Typical Year).
- **Label 0 (Less Suitable):** Maps to the **10th Percentile** (Worst Case/Crop Failure).

### 🛠️ Example ROI Calculation:
If a farmer asks for Paddy in a "Dry Monsoon" year:
1.  **XGBoost** predicts `Label 0` (Less Suitable) based on poor `rain_growth`.
2.  **System** pulls the **10th Percentile Yield** (~8-10 Q/Acre).
3.  **Result:** The **ROI on screen drops to -5%**, warning the farmer that they might lose money this season!

---

## 🧬 Deployment Step (When Ready):
1.  **Data Preprocessing:** Modify `SuitabilityModel.train` to calculate these 3-phase averages and save the 90/50/10 yield percentiles per crop.
2.  **Retraining:** Run `SuitabilityModel.train()` once to bake these temporal patterns into the JSON/XGB model.
3.  **API Integration:** Update [hybrid_recommender.py](file:///c:/Users/abhir/Desktop/Final%20Project/src/hybrid_recommender.py) to use `yield_profile[label]` instead of a fixed average.
