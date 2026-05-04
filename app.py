"""
app.py — AgriSetu Flask API Server
===================================
Wraps the hybrid recommender backend into a REST API and serves
the agri-setu-master frontend as static files.

Run from project root:
    python app.py

Then open:  http://127.0.0.1:5000/
"""

import os
import sys
import json
import logging

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(BASE_DIR, "src")
STATIC_DIR = os.path.join(BASE_DIR, "agri-setu-master")

sys.path.insert(0, SRC_DIR)

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, send_from_directory

# Silence noisy loggers before importing heavy deps
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# ── Enable CORS for all responses (needed for file:// dev and cross-origin)
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Lazy-load heavy models so Flask starts fast ────────────────────
_hr_module = None
_suitability_db = None

def get_models():
    global _hr_module, _suitability_db
    if _hr_module is None:
        import warnings
        warnings.filterwarnings("ignore")
        import importlib
        import hybrid_recommender as hr
        import suitability_model
        _hr_module = hr
        _suitability_db = suitability_model.CropSuitabilityModel()
    return _hr_module, _suitability_db


# ── Helper: run recommender and collect results as a list of dicts ───────────
def run_recommendation(district: str, budget: float, duration_months: float, soil_type: str) -> list:
    """
    Runs the hybrid recommender and returns a list of ranked crop dicts.
    Refactored to use the central hybrid_recommender.py logic.
    """
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    hr, s_db = get_models()

    # 1. Call the Central Source of Truth (Pass pre-loaded model for speed)
    final_df = hr.hybrid_recommendation(district, budget, duration_months, soil_type, suitability_db=s_db)
    
    if final_df is None or final_df.empty:
        return [], {}

    # 2. Use Historical Weather Data (No more live API calls for AI logic)
    import weather_service
    weather_ctx = weather_service.get_historical_weather_forecast(district)
    weather_info = {
        "season":   weather_ctx.get("season", "N/A"),
        "avg_temp": round(weather_ctx.get("avg_temp", 27.5), 1),
        "avg_rain": round(weather_ctx.get("avg_rain", 5.0), 1),
    }

    # 3. Enrich the Results with extra UI metadata
    results = []
    for _, row in final_df.iterrows():
        crop = row["crop_name"]
        
        # Pull dynamic calculations directly from the AI DataFrame instead of re-calculating
        rev = row.get("revenue", 0)
        harv_cost = row.get("harvest_cost", 0)
        profit = row.get("profit_cycle", 0)
        total_cost = row.get("initial_cost_inr_per_acre", 0) + row.get("maintenance_cost_inr_per_acre", 0) + harv_cost
        dynamic_yield = row.get("dynamic_expected_yield", 0)
        
        results.append({
            "crop_name":        crop,
            "hybrid_score":     round(float(row["hybrid_score"]), 4),
            "est_roi":          round(float(row["est_roi"]), 1),
            "predicted_price":  round(float(row["predicted_price"]), 0),
            "volatility_index": round(float(row["volatility_index"]), 3),
            "suitability_score":round(float(row["suitability_score"]), 4),
            "recommendation_text": str(row.get("recommendation_text", "")),
            "risk_level":       "Low" if row["volatility_index"] < 0.15 else "High",
            "risk_note":        str(row.get("risk_note", "")),
            "companion_crops":  data_utils.COMPANION_MAP.get(crop, []),
            "initial_cost":     int(row["initial_cost_inr_per_acre"]),
            "approx_months":    round(float(row["approx_months"]), 1),
            "maint_cost":       int(row["maintenance_cost_inr_per_acre"]),
            "harvest_cost":     int(harv_cost),
            "revenue":          int(rev),
            "profit_cycle":     int(profit),
            "est_yield":        float(dynamic_yield),
            "annual_profit":    int(profit), # Kept for API compatibility
            "annual_cost":      int(total_cost)  # Kept for API compatibility
        })

    return results, weather_info


# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "home.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return jsonify({}), 200


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    POST body (JSON):
    {
        "district":        "Thiruvananthapuram",
        "budget":          50000,
        "duration_months": 6,
        "soil_type":       "Loamy"
    }

    Returns:
    {
        "success": true,
        "district": "...",
        "weather": {...},
        "recommendations": [ { crop_name, hybrid_score, est_roi, ... }, ... ]
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    district        = str(data.get("district", "Thiruvananthapuram")).strip()
    budget          = float(data.get("budget", 50000))
    duration_months = float(data.get("duration_months", 6))
    soil_type       = str(data.get("soil_type", "Loamy")).strip().capitalize()

    valid_soils = ["Clay", "Sandy", "Loamy", "Red", "Laterite"]
    if soil_type not in valid_soils:
        soil_type = "Loamy"

    try:
        recs, weather_info = run_recommendation(district, budget, duration_months, soil_type)
        if not recs:
            return jsonify({
                "success": False,
                "error":   "No crops matched your budget and duration constraints."
            }), 200

        return jsonify({
            "success":         True,
            "district":        district,
            "weather":         weather_info,
            "recommendations": recs
        })

    except Exception as e:
        app.logger.exception("Recommendation error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/top-recommendation", methods=["GET"])
def top_recommendation():
    """
    Returns the top-1 crop for the home page hero card.
    Uses a default query (Thiruvananthapuram, 50000, 6 months, Loamy).
    Accepts optional query params: ?district=...&budget=...&months=...&soil=...
    """
    district        = request.args.get("district", "Thiruvananthapuram")
    budget          = float(request.args.get("budget", 50000))
    duration_months = float(request.args.get("months", 6))
    soil_type       = request.args.get("soil", "Loamy").capitalize()

    try:
        recs, weather_info = run_recommendation(district, budget, duration_months, soil_type)
        top = recs[0] if recs else None
        return jsonify({
            "success": True,
            "district": district,
            "weather":  weather_info,
            "top":      top
        })
    except Exception as e:
        app.logger.exception("Top recommendation error")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriSetu Backend API")
    print("  Open:  http://127.0.0.1:5000/")
    print("  API:   http://127.0.0.1:5000/api/recommend  (POST)")
    print("         http://127.0.0.1:5000/api/top-recommendation  (GET)")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)
