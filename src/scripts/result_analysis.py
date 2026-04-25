"""
result_analysis.py
------------------
Generates visual analysis of the hybrid recommender weight optimization
and crop performance data. Saves charts to the project root.

Run from project root:
    python src/result_analysis.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
# Because this script is in src/scripts/, BASE_DIR is two directories up
PARENT_DIR = os.path.dirname(SRC_DIR)
BASE_DIR = os.path.dirname(PARENT_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
sys.path.insert(0, PARENT_DIR) # Add src/ to sys.path

import data_utils
import suitability_model as sm

OUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "Banana":  "#f4d03f",
    "Coconut": "#27ae60",
    "Cashew":  "#e67e22",
    "Coffee":  "#6e2f1a",
    "Rice":    "#85c1e9",
    "Tapioca": "#af7ac5",
}

CROP_FILES = {
    "Coconut": os.path.join(DATA_DIR, "cocunut_data.csv"),
    "Cashew":  os.path.join(DATA_DIR, "cashew_data.csv"),
    "Coffee":  os.path.join(DATA_DIR, "coffee_data.csv"),
    "Rice":    os.path.join(DATA_DIR, "rice_data.csv"),
    "Banana":  os.path.join(DATA_DIR, "banana_data.csv"),
    "Tapioca": os.path.join(DATA_DIR, "tapioca_data.csv"),
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def style():
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#3a3f5c",
        "axes.labelcolor":  "#c8cde4",
        "xtick.color":      "#c8cde4",
        "ytick.color":      "#c8cde4",
        "text.color":       "#c8cde4",
        "grid.color":       "#2a2f45",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "font.family":      "sans-serif",
        "font.size":        11,
    })

def load_crop_aggregates(crop_attrs):
    cost_map = crop_attrs.set_index("crop_name")[
        ["initial_cost_inr_per_acre","maintenance_cost_inr_per_acre"]
    ].to_dict("index")

    rows = []
    for crop, path in CROP_FILES.items():
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df["p_modal"]         = pd.to_numeric(df["p_modal"],         errors="coerce")
        df["yield_kg_per_ha"] = pd.to_numeric(df["yield_kg_per_ha"], errors="coerce")
        df = df.dropna(subset=["p_modal","yield_kg_per_ha"])
        df = df[df["p_modal"] > 0]
        if df.empty: continue

        costs = None
        for n in [crop, crop.capitalize(), crop.title()]:
            if n in cost_map: costs = cost_map[n]; break
        if costs is None: continue

        total_cost = costs["initial_cost_inr_per_acre"] + costs["maintenance_cost_inr_per_acre"]
        avg_price  = df["p_modal"].mean()
        avg_yield  = df["yield_kg_per_ha"].mean()
        yield_q    = avg_yield / 247.105 / 100
        revenue    = yield_q * avg_price
        profit     = revenue - total_cost
        roi        = profit / total_cost * 100

        rows.append(dict(crop=crop, avg_price=avg_price, avg_yield_kg_ha=avg_yield,
                         total_cost=total_cost, revenue=revenue, profit=profit, roi=roi))
    return pd.DataFrame(rows)


def get_suitability(agg_df):
    model = sm.CropSuitabilityModel()
    # Use Thiruvananthapuram as a representative district
    result = model.predict_suitability("Thiruvananthapuram")
    suit_map = result.set_index("crop_name")["suitability_score"].to_dict()
    agg_df["suitability"] = agg_df["crop"].map(suit_map)
    return agg_df


def normalize(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi != lo else s * 0 + 1


# ── Chart 1: Old vs Learned Weights ────────────────────────────────────────

def plot_weights(ax, learned):
    old    = {"Suitability": 0.50, "Price": 0.30, "ROI": 0.20}
    new    = {"Suitability": learned["w_suit"], "Price": learned["w_price"], "ROI": learned["w_roi"]}
    labels = list(old.keys())
    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w/2, [old[l] for l in labels],  w, label="Hardcoded", color="#4a6fa5", alpha=0.85)
    bars2 = ax.bar(x + w/2, [new[l] for l in labels],  w, label="Learned",   color="#e74c3c", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Weight value"); ax.set_title("Scoring Weights: Hardcoded vs Learned", fontsize=13, pad=10)
    ax.set_ylim(0, 0.75); ax.legend(fontsize=10); ax.grid(axis="y")

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#e74c3c")


# ── Chart 2: Profit per Acre by Crop ───────────────────────────────────────

def plot_profit(ax, agg):
    agg_sorted = agg.sort_values("profit", ascending=True)
    colors = [PALETTE.get(c, "#7f8c8d") for c in agg_sorted["crop"]]
    bars = ax.barh(agg_sorted["crop"], agg_sorted["profit"]/1000, color=colors, alpha=0.88)
    ax.set_xlabel("Avg Profit per Acre (₹ thousands)")
    ax.set_title("Crop Profitability (Avg Market Price × Yield − Cost)", fontsize=13, pad=10)
    ax.axvline(0, color="#e74c3c", linewidth=1, linestyle="--")
    ax.grid(axis="x")
    for bar, val in zip(bars, agg_sorted["profit"]/1000):
        ax.text(val + (0.5 if val >= 0 else -2), bar.get_y()+bar.get_height()/2,
                f"₹{val:,.1f}k", va="center", fontsize=9)


# ── Chart 3: Yield vs Average Price bubble chart ────────────────────────────

def plot_yield_price(ax, agg):
    size_scale = 2000
    for _, row in agg.iterrows():
        c = PALETTE.get(row["crop"], "#7f8c8d")
        ax.scatter(row["avg_yield_kg_ha"], row["avg_price"],
                   s=abs(row["profit"])/500 + 100,
                   color=c, alpha=0.85, edgecolors="white", linewidth=0.8)
        ax.annotate(row["crop"],
                    (row["avg_yield_kg_ha"], row["avg_price"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=c)

    ax.set_xlabel("Avg Yield (kg / ha)"); ax.set_ylabel("Avg Market Price (₹ / quintal)")
    ax.set_title("Yield vs. Market Price\n(bubble size = |profit|)", fontsize=13, pad=10)
    ax.grid(True)


# ── Chart 4: Hybrid Score Comparison Old vs New Weights ─────────────────────

def plot_score_comparison(ax, agg, learned):
    agg = agg.copy()
    agg["ns"] = normalize(agg["suitability"])
    agg["np"] = normalize(agg["avg_price"])
    agg["nr"] = normalize(agg["roi"])

    old_w  = (0.50, 0.30, 0.20)
    new_w  = (learned["w_suit"], learned["w_price"], learned["w_roi"])

    agg["score_old"] = old_w[0]*agg["ns"] + old_w[1]*agg["np"] + old_w[2]*agg["nr"]
    agg["score_new"] = new_w[0]*agg["ns"] + new_w[1]*agg["np"] + new_w[2]*agg["nr"]

    agg = agg.sort_values("score_new", ascending=False)
    x   = np.arange(len(agg))
    w   = 0.35

    ax.bar(x - w/2, agg["score_old"], w, label="Old weights (0.5/0.3/0.2)", color="#4a6fa5", alpha=0.85)
    ax.bar(x + w/2, agg["score_new"], w, label="Learned weights", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(agg["crop"], rotation=15, fontsize=10)
    ax.set_ylabel("Hybrid Score (0–1)"); ax.set_title("Hybrid Score: Old vs Learned Weights", fontsize=13, pad=10)
    ax.legend(fontsize=9); ax.grid(axis="y"); ax.set_ylim(0, 1.15)

    # Annotate rank changes
    old_rank = agg.sort_values("score_old", ascending=False).reset_index()["crop"].tolist()
    new_rank = agg.reset_index()["crop"].tolist()
    for i, crop in enumerate(new_rank):
        old_r = old_rank.index(crop) + 1
        new_r = i + 1
        delta = old_r - new_r
        if delta != 0:
            arrow = "▲" if delta > 0 else "▼"
            col   = "#2ecc71" if delta > 0 else "#e74c3c"
            ax.text(x[i] + w/2, agg[agg["crop"]==crop]["score_new"].values[0] + 0.04,
                    f"{arrow}{abs(delta)}", ha="center", color=col, fontsize=9, fontweight="bold")


# ── Chart 5: ROI by crop ─────────────────────────────────────────────────────

def plot_roi(ax, agg):
    agg_sorted = agg.sort_values("roi", ascending=True)
    colors = [PALETTE.get(c, "#7f8c8d") for c in agg_sorted["crop"]]
    bars = ax.barh(agg_sorted["crop"], agg_sorted["roi"], color=colors, alpha=0.88)
    ax.set_xlabel("ROI (%)")
    ax.set_title("Return on Investment per Crop", fontsize=13, pad=10)
    ax.axvline(0, color="#e74c3c", linewidth=1, linestyle="--")
    ax.grid(axis="x")
    for bar, val in zip(bars, agg_sorted["roi"]):
        ax.text(val + 2, bar.get_y()+bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=9)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    style()

    print("Loading data ...")
    crop_attrs = pd.read_csv(os.path.join(DATA_DIR, "crop_attributes.csv"))
    learned    = json.load(open(os.path.join(MODELS_DIR, "learned_weights.json")))

    agg = load_crop_aggregates(crop_attrs)
    agg = get_suitability(agg)
    agg = agg.dropna()

    print(f"  {len(agg)} crops with complete data: {list(agg['crop'])}")

    # ── Full dashboard ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor="#0f1117")
    fig.suptitle("Hybrid Recommender — Result Analysis Dashboard",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])

    plot_weights(ax1, learned)
    plot_profit(ax2, agg)
    plot_yield_price(ax3, agg)
    plot_score_comparison(ax4, agg, learned)
    plot_roi(ax5, agg)

    out_path = os.path.join(OUT_DIR, "analysis_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\nDashboard saved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Crop':<10} {'Avg Price':>12} {'Avg Yield (kg/ha)':>18} {'ROI %':>8} {'Profit ₹':>12}")
    print("-"*65)
    for _, r in agg.sort_values("profit", ascending=False).iterrows():
        print(f"{r['crop']:<10} {r['avg_price']:>12,.0f} {r['avg_yield_kg_ha']:>18,.1f} {r['roi']:>8.1f} {r['profit']:>12,.0f}")
    print("="*65)

    print(f"\nLearned weights: suit={learned['w_suit']:.3f}  price={learned['w_price']:.3f}  roi={learned['w_roi']:.3f}")


if __name__ == "__main__":
    main()
