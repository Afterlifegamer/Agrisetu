import sys
sys.path.insert(0, '../src')

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

import hybrid_recommender as hr

scenarios = [
    ("Thiruvananthapuram", 50000,  12, "Loamy"),
    ("Thiruvananthapuram", 80000,   9, "Red"),
    ("Thiruvananthapuram", 50000,  60, "Loamy"),
    ("Thiruvananthapuram", 70000,  36, "Red"),
    ("Thiruvananthapuram", 40000,  12, "Clay"),
    ("Thiruvananthapuram", 100000,  6, "Loamy"),
]

for district, budget, duration, soil in scenarios:
    print(f"\n{'='*60}")
    print(f"Budget=₹{budget}  Duration={duration}mo  Soil={soil}")
    print('='*60)
    df = hr.hybrid_recommendation(district, budget, duration, soil)
    if df is not None and not df.empty:
        top = df[['crop_name','est_roi','dynamic_expected_yield']].head(5)
        print(top.to_string(index=False))
    else:
        print("No crops matched.")
