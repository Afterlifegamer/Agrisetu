import pandas as pd
import numpy as np
import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class CropSuitabilityModel:
    def __init__(self, weather_file=None, crop_file=None, model_file=None):
        # Base Paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        MODELS_DIR = os.path.join(BASE_DIR, 'models')

        self.weather_file = weather_file or os.path.join(DATA_DIR, "kerala_weather_2023_2024_full.csv")
        self.crop_file = crop_file or os.path.join(DATA_DIR, "crop_attributes.csv")
        self.model_file = model_file or os.path.join(MODELS_DIR, "suitability_xgb.json")
        self.yield_profiles_file = os.path.join(MODELS_DIR, "crop_yield_profiles.json")
        
        # Load Data
        self.weather_data = pd.read_csv(self.weather_file)
        self.crop_data = pd.read_csv(self.crop_file)
        
        # Determine global average rainfall, temp, hum
        self.weather_data['date'] = pd.to_datetime(self.weather_data['date'])
        self.global_avg_rain = self.weather_data["rain_mm"].mean()
        self.global_avg_temp = self.weather_data["temp_2m_C"].mean()
        self.global_avg_hum = self.weather_data["rh_2m_percent"].mean()

        # Build Climatology Calendar (1-12 month averages)
        self.climatology_rain = self.weather_data.groupby(self.weather_data['date'].dt.month)['rain_mm'].mean().to_dict()
        self.climatology_temp = self.weather_data.groupby(self.weather_data['date'].dt.month)['temp_2m_C'].mean().to_dict()
        self.climatology_hum = self.weather_data.groupby(self.weather_data['date'].dt.month)['rh_2m_percent'].mean().to_dict()

        # Pre-process mappings
        self._prepare_mappings()
        
        # LOAD or TRAIN
        if os.path.exists(self.model_file) and os.path.exists(self.yield_profiles_file):
            self.model = XGBClassifier()
            self.model.load_model(self.model_file)
        else:
            print("⚠️ Saved model or yield profiles not found. Training from scratch...")
            self.train()
            
    def _prepare_mappings(self):
        water_map = {"Low": 1, "Medium": 2, "High": 3}
        effort_map = {"Low": 1, "Medium": 2, "High": 3}
        self.crop_data["water_dependency_num"] = self.crop_data["water_dependency"].map(water_map).fillna(2)
        self.crop_data["time_effort_num"] = self.crop_data["time_effort"].map(effort_map).fillna(2)

    def _calculate_suitability(self, row, rainfall):
        # Fallback expert simulator in case of missing data
        water_dep = row["water_dependency_num"] if row["water_dependency_num"] > 0 else 2
        ideal_map  = {1: 2.0, 2: 5.0, 3: 10.0}
        ideal_rainfall = ideal_map.get(int(water_dep), 5.0)

        sigma = 1.5
        water_score = np.exp(-((rainfall - ideal_rainfall) ** 2) / (2 * sigma ** 2))
        effort = row["time_effort_num"] if row["time_effort_num"] > 0 else 1
        effort_score = 1.0 / effort
        suitability = (water_score * 0.70) + (effort_score * 0.30)
        return suitability

    def _get_historical_price(self, crop_name, target_year, target_month):
        import sys
        import data_utils
        base_price = data_utils.BASE_PRICES.get(crop_name.title(), 5000)
        
        # Resolve filenames: crop_price.csv or crop_price_data.csv
        data_dir = os.path.dirname(self.weather_file)
        new_dir = os.path.join(data_dir, "New")
        p1 = os.path.join(new_dir, f"{crop_name.lower()}_price.csv")
        p2 = os.path.join(new_dir, f"{crop_name.lower()}_price_data.csv")
        
        target_file = None
        if os.path.exists(p1):
            target_file = p1
        elif os.path.exists(p2):
            target_file = p2
        else:
            return base_price
            
        try:
            df = pd.read_csv(target_file)
            df['t'] = pd.to_datetime(df['t'])
            mask = (df['t'].dt.year == target_year) & (df['t'].dt.month == target_month)
            matched = df[mask]
            if not matched.empty:
                vals = pd.to_numeric(matched['p_modal'], errors='coerce').dropna()
                if not vals.empty: return vals.mean()
            
            vals = pd.to_numeric(df['p_modal'], errors='coerce').dropna()
            if not vals.empty: return vals.mean()
            return base_price
        except:
            return base_price

    def _get_phase_stats(self, anchor_month, duration_months, direction='forward'):
        months = []
        if direction == 'forward':
            for i in range(duration_months):
                months.append(((anchor_month - 1 + i) % 12) + 1)
        else: # backward (from harvest month)
            for i in range(duration_months):
                months.append(((anchor_month - 1 - i) % 12) + 1)
            months.reverse() # ensure Sowing -> Growth -> Harvest order

        p1_end = max(1, round(duration_months * 0.25))
        p2_end = max(p1_end + 1, round(duration_months * 0.75))
        if p2_end >= duration_months: p2_end = duration_months - 1
        if p1_end >= p2_end: p1_end = max(1, p2_end - 1)
        
        sowing_months = months[:p1_end]
        growth_months = months[p1_end:p2_end]
        harvest_months = months[p2_end:]
        
        if not sowing_months: sowing_months = [months[0]]
        if not growth_months: growth_months = [months[len(months)//2]]
        if not harvest_months: harvest_months = [months[-1]]
        
        stats = {}
        for phase, phase_months in [("sowing", sowing_months), ("growth", growth_months), ("harvest", harvest_months)]:
            stats[f"rain_{phase}"] = np.mean([self.climatology_rain.get(m, self.global_avg_rain) for m in phase_months])
            stats[f"temp_{phase}"] = np.mean([self.climatology_temp.get(m, self.global_avg_temp) for m in phase_months])
            stats[f"hum_{phase}"] = np.mean([self.climatology_hum.get(m, self.global_avg_hum) for m in phase_months])
            
        return stats

    def train(self):
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import data_utils
        
        yield_file = os.path.join(os.path.dirname(self.weather_file), "kerala_monthly_estimates_2023_2025.csv")
        try:
            yield_df = pd.read_csv(yield_file)
        except Exception as e:
            print(f"Error loading yield data: {e}")
            return
            
        training_rows = []
        yield_profiles = {}
        
        # Cross-Map Equivalent Crops for Training
        crop_aliases = {"Rice": "Paddy"}

        for index, row in self.crop_data.iterrows():
            crop_name = str(row['crop_name']).strip()
            lookup_name = crop_aliases.get(crop_name, crop_name)
            
            duration_days = data_utils.parse_duration_to_days(row['duration_to_harvest'])
            duration_months = max(1, round(duration_days / 30.0))
            
            crop_yields = yield_df[yield_df['Crop'].str.lower() == lookup_name.lower()]
            
         
            # Convert kg/ha to Quintals/Acre (divide by 247.105)
            prod_col = 'Estimated_Productivity_kg_per_ha'
            if prod_col in crop_yields.columns and not crop_yields[prod_col].dropna().empty:
                valid_prod = crop_yields[prod_col].dropna()
                
                # 10 Continuous-Like Decile Buckets
                yield_profiles[crop_name] = {}
                for i in range(10):
                    percentile = (i + 1) * 0.10  # 0.1 to 1.0
                    val = valid_prod.quantile(percentile) / 247.105
                    yield_profiles[crop_name][str(i)] = round(val, 2)
            else:
                # Fallback to 10 linear steps (0.5x to 1.5x of baseline)
                baseline = data_utils.YIELD_ESTIMATES.get(crop_name, 10.0)
                yield_profiles[crop_name] = {}
                for i in range(10):
                    multiplier = 0.5 + (0.111 * i)
                    yield_profiles[crop_name][str(i)] = round(baseline * multiplier, 2)
                
            pos_yields = crop_yields[crop_yields['Monthly_Production_MT'] > 0]['Monthly_Production_MT']
            deciles = []
            if not pos_yields.empty:
                deciles = [pos_yields.quantile((i + 1) * 0.10) for i in range(10)]
                
            for _, y_row in crop_yields.iterrows():
                harvest_month = int(y_row['Month'])
                harvest_year = int(y_row['Year'])
                volume = float(y_row['Monthly_Production_MT'])
                
                from dateutil.relativedelta import relativedelta
                import datetime
                harvest_date = datetime.datetime(harvest_year, harvest_month, 1)
                planting_date = harvest_date - relativedelta(months=duration_months)
                
                price_at_planting = self._get_historical_price(crop_name, planting_date.year, planting_date.month)
                
                label = 0
                if volume > 0 and deciles:
                    # Find exactly which of the 10 buckets this historical yield belongs to
                    for idx, boundary in enumerate(deciles):
                        if volume <= boundary:
                            label = idx
                            break
                    else:
                        label = 9
                    
                phase_stats = self._get_phase_stats(harvest_month, duration_months, direction='backward')
                
                row_data = {
                    "crop_name": crop_name,
                    "water_dependency_num": row["water_dependency_num"],
                    "time_effort_num": row["time_effort_num"],
                    "price_at_planting": price_at_planting,
                    "suitability_label": label
                }
                row_data.update(phase_stats)
                training_rows.append(row_data)
                
        if not training_rows:
            print("No training data generated.")
            return

        with open(self.yield_profiles_file, 'w') as f:
            json.dump(yield_profiles, f, indent=4)
            
        training_data = pd.DataFrame(training_rows)
        X = training_data[[
            "rain_sowing", "temp_sowing", "hum_sowing",
            "rain_growth", "temp_growth", "hum_growth",
            "rain_harvest", "temp_harvest", "hum_harvest",
            "water_dependency_num", "time_effort_num", "price_at_planting"
        ]]
        y = training_data["suitability_label"]

        import optuna
        from sklearn.model_selection import train_test_split
        
        print("\nStarting Bayesian Hyperparameter Optimization with Optuna...")
        # Split data for early stopping evaluation (20% for validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            # 1. Define hyperparameter search space
            param = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                # Set a high number of trees, Early Stopping will cut it off optimally
                'n_estimators': 1500,
                'early_stopping_rounds': 15,
                'random_state': 42
            }
            
            # 2. Train the model with early stopping
            model = XGBClassifier(**param)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # 3. Evaluate accuracy
            preds = model.predict(X_val)
            accuracy = (preds == y_val).mean()
            
            # Track the optimal n_estimators found by early stopping
            if hasattr(model, 'best_iteration'):
                trial.set_user_attr("best_n_estimators", getattr(model, 'best_iteration', 100))
            else:
                trial.set_user_attr("best_n_estimators", 100)
                
            return accuracy
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30) # Train 30 combinations intelligently
        
        best_params = study.best_params
        best_n_estimators = study.best_trial.user_attrs.get("best_n_estimators", 100)
        
        print(f"\nOptimization Finished!")
        print(f"Best Depth/Learning Parameters: {best_params}")
        print(f"Optimal Number of Trees (via Early Stopping): {best_n_estimators}\n")
        
        # -------------------------------------------------------------
        # FINAL MODEL TRAINING
        # -------------------------------------------------------------
        # Retrain the final model on the FULL dataset using the discovered optimal parameters
        final_params = best_params.copy()
        final_params['n_estimators'] = best_n_estimators
        final_params['random_state'] = 42

        self.model = XGBClassifier(**final_params)
        self.model.fit(X, y)
        self.model.save_model(self.model_file)
        print("Model trained organically on Phase-Based Climatology + Yield Data with Optimized Hyperparameters and saved.")

    def get_season(self, rainfall):
        if rainfall > 10: return "High Rainfall"
        elif rainfall > 5: return "Moderate Rainfall"
        else: return "Low Rainfall"

    def predict_suitability(self, district_name, current_month=None):
        import datetime
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import data_utils
        
        if current_month is None:
            current_month = datetime.datetime.now().month
            
        district_weather = self.weather_data[self.weather_data['district'] == district_name].copy()
        if district_weather.empty:
            recent_avg_rainfall = self.global_avg_rain
        else:
            recent_avg_rainfall = district_weather.tail(30)["rain_mm"].mean()

        forecast_rainfall = recent_avg_rainfall * 1.5
        season = self.get_season(forecast_rainfall)

        results = self.crop_data.copy()
        pred_rows = []
        # Construct Future Weather Forecasts specific to each crop's biological gestation length
        for index, row in results.iterrows():
            duration_days = data_utils.parse_duration_to_days(row['duration_to_harvest'])
            duration_months = max(1, round(duration_days / 30.0))
            
            phase_stats = self._get_phase_stats(current_month, duration_months, direction='forward')
            
            # THE CONTROL VARIABLE TRICK:
            # Feed a fake neutral price to force the AI to ignore the Economy.
            baseline_price = data_utils.BASE_PRICES.get(row['crop_name'].title(), 5000)
            
            pred_row = {
                "water_dependency_num": row["water_dependency_num"],
                "time_effort_num": row["time_effort_num"],
                "price_at_planting": baseline_price
            }
            pred_row.update(phase_stats)
            pred_rows.append(pred_row)
            
        X_pred = pd.DataFrame(pred_rows)[
            ["rain_sowing", "temp_sowing", "hum_sowing",
             "rain_growth", "temp_growth", "hum_growth",
             "rain_harvest", "temp_harvest", "hum_harvest",
             "water_dependency_num", "time_effort_num", "price_at_planting"]
        ]
        
        try:
            probs = self.model.predict_proba(X_pred)
            continuous_scores = []
            predicted_labels = []
            for i, p in enumerate(probs):
                # We now expect 10 buckets out of the XGBClassifier!
                # Create a continuous 0.0 - 1.0 Expectation Ratio from the 10 probabilities:
                expected_ratio = sum(prob * (label_idx / 9.0) for label_idx, prob in enumerate(p))
                
                # FUSE THE ENSEMBLE ENGINE: 70% AI (Future N-month), 30% Physics (Next 30 days)
                row = results.iloc[i]
                physics_score = self._calculate_suitability(row, forecast_rainfall)
                
                # The blended physics/AI score tells us exactly which Bucket (0-9) to snap to!
                blended_ratio = (expected_ratio * 0.70) + (physics_score * 0.30)
                final_bucket = int(round(blended_ratio * 9))
                
                # Ensure the bucket is mathematically bounded 0-9
                final_bucket = max(0, min(9, final_bucket))
                
                continuous_scores.append(blended_ratio)
                predicted_labels.append(final_bucket)
            results["suitability_score"] = continuous_scores
            results["predicted_label"] = predicted_labels
        except Exception as e:
            print(f"Prediction failed, using fallback: {e}")
            physics_scores = results.apply(lambda r: self._calculate_suitability(r, forecast_rainfall), axis=1)
            results["suitability_score"] = physics_scores
            results["predicted_label"] = physics_scores.apply(lambda x: max(0, min(9, int(round(x * 9)))))
            
        # UI Interpretation mapping for the new 10-Tier ranking system
        label_map = {i: f"Tier {i+1} Biological Forecast" for i in range(10)}
        results["recommendation_text"] = results["predicted_label"].map(label_map).fillna("Unknown")
        results["forecast_season"] = season
        results["forecast_rainfall"] = forecast_rainfall
        
        return results[[
            "crop_name", 
            "suitability_score", 
            "predicted_label",
            "recommendation_text", 
            "initial_cost_inr_per_acre", 
            "maintenance_cost_inr_per_acre",
            "duration_to_harvest",
            "forecast_season"
        ]]

if __name__ == "__main__":
    pass
