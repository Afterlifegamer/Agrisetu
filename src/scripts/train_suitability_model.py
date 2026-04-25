from suitability_model import CropSuitabilityModel
import os

def train_and_save_model():
    print("🚀 Starting training for Crop Suitability Model...")
    
    # 1. Initialize Model 
    # (This will automatically load if exists, but we want to force train)
    model = CropSuitabilityModel()
    
    # 2. Force Retrain
    print("   Training XGBoost Classifier...")
    model.train()
    
    # 3. Verify
    print("   ✅ Suitability model trained and saved.")
    
    # Optional: Quick Test
    print("\n🔎 Running Quick Test for 'Thiruvananthapuram'...")
    try:
        results = model.predict_suitability("Thiruvananthapuram")
        print(results.head(3))
    except Exception as e:
        print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    train_and_save_model()
