"""
Hyperparameter test script - runs GridSearchCV on the suitability model training data
and reports train/test accuracy to determine if tuning is worthwhile.
"""
import sys, os, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import data_utils, suitability_model as sm

# ── Build training data by replicating the train() logic exactly ────────────
model_obj = sm.CropSuitabilityModel()
yield_file = os.path.join("data", "kerala_monthly_estimates_2023_2025.csv")
yield_df   = pd.read_csv(yield_file)
crop_aliases = {"Rice": "Paddy"}
training_rows = []

for index, row in model_obj.crop_data.iterrows():
    crop_name  = str(row['crop_name']).strip()
    lookup     = crop_aliases.get(crop_name, crop_name)
    dur_days   = data_utils.parse_duration_to_days(row['duration_to_harvest'])
    dur_months = max(1, round(dur_days / 30.0))
    crop_yields = yield_df[yield_df['Crop'].str.lower() == lookup.lower()]

    prod_col = 'Estimated_Productivity_kg_per_ha'
    if prod_col in crop_yields.columns and not crop_yields[prod_col].dropna().empty:
        vals = crop_yields[prod_col].dropna() / 247.105
        deciles = [float(np.quantile(vals, q)) for q in np.arange(0.1, 1.01, 0.1)]
    else:
        base = data_utils.YIELD_ESTIMATES.get(crop_name, 10)
        deciles = [base * f for f in np.linspace(0.5, 1.5, 10)]

    for _, yrow in crop_yields.iterrows():
        try:
            harvest_month = int(str(yrow.get('Month', 6)))
            volume = float(yrow.get(prod_col, 0)) / 247.105
            price_at_planting = model_obj._get_historical_price(
                crop_name, harvest_month, 2024,
                data_utils.BASE_PRICES.get(crop_name, 5000))
        except:
            continue

        label = 0
        if volume > 0 and deciles:
            for idx, boundary in enumerate(deciles):
                if volume <= boundary:
                    label = idx
                    break
            else:
                label = 9

        phase_stats = model_obj._get_phase_stats(harvest_month, dur_months, 'backward')
        row_data = {
            "water_dependency_num": float(row["water_dependency_num"]),
            "time_effort_num":      float(row["time_effort_num"]),
            "price_at_planting":    float(price_at_planting),
            "suitability_label":    label
        }
        row_data.update(phase_stats)
        training_rows.append(row_data)

df = pd.DataFrame(training_rows)
FEATURES = [
    "rain_sowing","temp_sowing","hum_sowing",
    "rain_growth","temp_growth","hum_growth",
    "rain_harvest","temp_harvest","hum_harvest",
    "water_dependency_num","time_effort_num","price_at_planting"
]
X = df[FEATURES]
y = df["suitability_label"]

print(f"\nDataset size: {len(df)} rows, {y.nunique()} unique buckets")
print(f"Bucket distribution:\n{y.value_counts().sort_index()}\n")

# ── Baseline ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
baseline = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0)
baseline.fit(X_train, y_train)
train_acc = accuracy_score(y_train, baseline.predict(X_train))
test_acc  = accuracy_score(y_test,  baseline.predict(X_test))
print(f"=== Baseline (n_estimators=100, max_depth=4) ===")
print(f"  Train accuracy: {train_acc:.2%}")
print(f"  Test  accuracy: {test_acc:.2%}")
gap = train_acc - test_acc
if gap > 0.20:
    print(f"  WARNING: OVERFITTING  (train-test gap={gap:.2%})")
elif test_acc < 0.40:
    print(f"  WARNING: UNDERFITTING (test={test_acc:.2%})")
else:
    print(f"  OK: Reasonable fit (gap={gap:.2%})")

# ── Grid Search ──────────────────────────────────────────────────────────────
print("\n=== Running GridSearchCV (27 combos x 5 folds) ... ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 4, 6],
    'learning_rate':[0.05, 0.1, 0.2],
}
gs = GridSearchCV(XGBClassifier(verbosity=0), param_grid, cv=5,
                  scoring='accuracy', n_jobs=-1)
gs.fit(X, y)

best = gs.best_estimator_
best_test = accuracy_score(y_test, best.predict(X_test))
improvement = best_test - test_acc
print(f"\nBest params:   {gs.best_params_}")
print(f"Best CV score: {gs.best_score_:.2%}")
print(f"Best test acc: {best_test:.2%}  (baseline was {test_acc:.2%})")
print(f"Improvement:   {improvement:+.2%}")
if improvement > 0.05:
    print("\nVERDICT: Worth implementing.")
else:
    print("\nVERDICT: Not worth implementing - improvement is negligible.")
