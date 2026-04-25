"""
model_comparison.py
-------------------
Evaluates 4 models against actual historical profitability:
  1. Suitability-only
  2. Price-only
  3. ROI-only  
  4. Hybrid (learned weights)

For each distinct district in the price CSVs, ranks crops using each
model and measures Spearman rank correlation with actual profit ranking.

Saves charts to: analysis/model_comparison.png
"""

import os, sys, json, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
# Because this script is in src/scripts/, BASE_DIR is two directories up
PARENT_DIR = os.path.dirname(SRC_DIR)
BASE_DIR = os.path.dirname(PARENT_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUT_DIR     = os.path.join(BASE_DIR, "analysis")
sys.path.insert(0, PARENT_DIR) # Add src/ to sys.path
os.makedirs(OUT_DIR, exist_ok=True)

import suitability_model as sm

CROP_FILES = {
    "Coconut": os.path.join(DATA_DIR, "cocunut_data.csv"),
    "Cashew":  os.path.join(DATA_DIR, "cashew_data.csv"),
    "Coffee":  os.path.join(DATA_DIR, "coffee_data.csv"),
    "Rice":    os.path.join(DATA_DIR, "rice_data.csv"),
    "Banana":  os.path.join(DATA_DIR, "banana_data.csv"),
    "Tapioca": os.path.join(DATA_DIR, "tapioca_data.csv"),
}

MODEL_COLORS = {
    "Suitability-only": "#3498db",
    "Price-only":        "#f39c12",
    "ROI-only":          "#9b59b6",
    "Hybrid":            "#2ecc71",
}

# ── Load & aggregate data ───────────────────────────────────────────────────

def load_data():
    crop_attrs = pd.read_csv(os.path.join(DATA_DIR, "crop_attributes.csv"))
    cost_map = crop_attrs.set_index("crop_name")[
        ["initial_cost_inr_per_acre","maintenance_cost_inr_per_acre"]
    ].to_dict("index")

    learned = json.load(open(os.path.join(MODELS_DIR, "learned_weights.json")))

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

        agg = (df.groupby("district_name")
                 .agg(avg_price=("p_modal","mean"),
                      avg_yield=("yield_kg_per_ha","mean"),
                      price_std=("p_modal","std"),
                      n_obs=("p_modal","count"))
                 .reset_index())
        agg["crop"]       = crop
        agg["total_cost"] = total_cost
        agg["yield_q"]    = agg["avg_yield"] / 247.105 / 100
        agg["revenue"]    = agg["yield_q"] * agg["avg_price"]
        agg["profit"]     = agg["revenue"] - agg["total_cost"]
        agg["roi"]        = agg["profit"] / agg["total_cost"] * 100
        rows.append(agg)

    data = pd.concat(rows, ignore_index=True)
    return data, learned


def add_suitability(data):
    print("  Computing suitability scores per district ...")
    model = sm.CropSuitabilityModel()
    districts = data["district_name"].dropna().unique()
    suit_rows = []
    for d in districts:
        try:
            r = model.predict_suitability(d)
            r["district_name"] = d
            suit_rows.append(r[["crop_name","district_name","suitability_score"]])
        except: pass
    if not suit_rows: return data
    suit_df = pd.concat(suit_rows).rename(columns={"crop_name":"crop"})
    return data.merge(suit_df, on=["crop","district_name"], how="left")


def normalize_col(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi != lo else pd.Series(np.ones(len(s)), index=s.index)


# ── Evaluate each model per district ───────────────────────────────────────

def evaluate(data, learned):
    results = []          # one row per district
    top1_correct = {m: 0 for m in MODEL_COLORS}
    total_districts = 0

    for district, grp in data.groupby("district_name"):
        if grp["crop"].nunique() < 3: continue   # need enough crops to rank
        grp = grp.dropna(subset=["suitability_score"]).copy()
        if len(grp) < 3: continue

        # Normalise features within district
        grp["n_suit"]  = normalize_col(grp["suitability_score"])
        grp["n_price"] = normalize_col(grp["avg_price"])
        grp["n_roi"]   = normalize_col(grp["roi"])

        # Ground truth: actual profit ranking
        true_rank = grp["profit"].rank(ascending=False)

        # Each model's predicted ranking
        scores = {
            "Suitability-only": grp["n_suit"],
            "Price-only":        grp["n_price"],
            "ROI-only":          grp["n_roi"],
            "Hybrid": (
                learned["w_suit"]  * grp["n_suit"] +
                learned["w_price"] * grp["n_price"] +
                learned["w_roi"]   * grp["n_roi"]
            )
        }

        row = {"district": district}
        best_profit_crop = grp.loc[grp["profit"].idxmax(), "crop"]

        for model_name, score_series in scores.items():
            pred_rank = score_series.rank(ascending=False)
            corr, pval = stats.spearmanr(true_rank, pred_rank)
            row[model_name] = corr

            # Top-1 accuracy: did model pick the most profitable crop?
            top1_crop = grp.loc[score_series.idxmax(), "crop"]
            if top1_crop == best_profit_crop:
                top1_correct[model_name] += 1

        results.append(row)
        total_districts += 1

    df_results = pd.DataFrame(results)
    top1_acc = {m: top1_correct[m] / total_districts * 100 for m in MODEL_COLORS}
    print(f"  Evaluated {total_districts} districts.")
    return df_results, top1_acc, total_districts


# ── Plot ────────────────────────────────────────────────────────────────────

def plot(df_results, top1_acc, total_districts, learned):
    plt.rcParams.update({
        "figure.facecolor":"#0f1117","axes.facecolor":"#1a1d27",
        "axes.edgecolor":"#3a3f5c","axes.labelcolor":"#c8cde4",
        "xtick.color":"#c8cde4","ytick.color":"#c8cde4",
        "text.color":"#c8cde4","grid.color":"#2a2f45",
        "grid.linestyle":"--","grid.alpha":0.5,
        "font.family":"sans-serif","font.size":11,
    })

    fig = plt.figure(figsize=(20, 13), facecolor="#0f1117")
    fig.suptitle("Hybrid Model vs Individual Models — Evaluation Against Historical Data",
                 fontsize=16, color="white", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    models = list(MODEL_COLORS.keys())
    colors = list(MODEL_COLORS.values())

    # ── Chart 1: Mean Spearman correlation per model ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    means = [df_results[m].mean() for m in models]
    stds  = [df_results[m].std()  for m in models]
    bars = ax1.bar(models, means, color=colors, alpha=0.85, width=0.5)
    ax1.errorbar(models, means, yerr=stds, fmt="none", color="white",
                 capsize=5, linewidth=1.5)
    ax1.set_title("Mean Spearman Rank Correlation\nwith Actual Profitability", fontsize=12)
    ax1.set_ylabel("Spearman ρ  (higher = better)")
    ax1.set_ylim(-0.3, 1.1); ax1.axhline(0, color="#e74c3c", linewidth=1, linestyle="--")
    ax1.grid(axis="y")
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f"{m:.3f}", ha="center", fontsize=10, fontweight="bold",
                 color="white")
    ax1.set_xticklabels(models, rotation=12, fontsize=9)

    # ── Chart 2: Top-1 Accuracy ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    accs = [top1_acc[m] for m in models]
    bars2 = ax2.bar(models, accs, color=colors, alpha=0.85, width=0.5)
    ax2.set_title(f"Top-1 Accuracy: Did Model Pick\nMost Profitable Crop? (n={total_districts} districts)",
                  fontsize=12)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 110); ax2.grid(axis="y")
    for bar, a in zip(bars2, accs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{a:.0f}%", ha="center", fontsize=11, fontweight="bold", color="white")
    ax2.set_xticklabels(models, rotation=12, fontsize=9)

    # ── Chart 3: Distribution of Spearman correlations (box plot) ─────────
    ax3 = fig.add_subplot(gs[0, 2])
    plot_data = [df_results[m].dropna().values for m in models]
    bp = ax3.boxplot(plot_data, patch_artist=True, notch=True,
                     medianprops={"color":"white","linewidth":2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for element in ["whiskers","caps","fliers"]:
        for item in bp[element]:
            item.set(color="#c8cde4", linewidth=1)
    ax3.set_xticks(range(1, len(models)+1))
    ax3.set_xticklabels(models, rotation=12, fontsize=9)
    ax3.set_title("Spearman ρ Distribution\nAcross All Districts", fontsize=12)
    ax3.set_ylabel("Spearman ρ"); ax3.grid(axis="y")
    ax3.axhline(0, color="#e74c3c", linewidth=1, linestyle="--")

    # ── Chart 4: Per-district correlation heatmap (top 20 districts) ───────
    ax4 = fig.add_subplot(gs[1, :2])
    top_districts = (df_results.set_index("district")[models]
                     .assign(hybrid_rank=lambda x: x["Hybrid"])
                     .sort_values("hybrid_rank", ascending=False)
                     .drop(columns="hybrid_rank")
                     .head(20))

    import matplotlib.cm as cm
    im = ax4.imshow(top_districts.values.T, aspect="auto",
                    cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax4.set_yticks(range(len(models))); ax4.set_yticklabels(models, fontsize=9)
    ax4.set_xticks(range(len(top_districts)))
    ax4.set_xticklabels(top_districts.index, rotation=45, ha="right", fontsize=7)
    ax4.set_title("Spearman ρ per District (top 20 by Hybrid score)  |  Green=High, Red=Low", fontsize=12)
    plt.colorbar(im, ax=ax4, orientation="horizontal", pad=0.25, fraction=0.03,
                 label="Spearman ρ")

    # Annotate cells
    for i, model in enumerate(models):
        for j, val in enumerate(top_districts[model]):
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color="black" if 0.2 < val < 0.8 else "white")

    # ── Chart 5: Improvement of Hybrid over best single model ─────────────
    ax5 = fig.add_subplot(gs[1, 2])
    best_single = df_results[["Suitability-only","Price-only","ROI-only"]].max(axis=1)
    improvement = df_results["Hybrid"] - best_single
    wins   = (improvement > 0).sum()
    losses = (improvement < 0).sum()
    ties   = (improvement == 0).sum()

    ax5.hist(improvement.dropna(), bins=20, color="#2ecc71", alpha=0.8, edgecolor="#0f1117")
    ax5.axvline(0, color="#e74c3c", linewidth=2, linestyle="--", label="No change")
    ax5.axvline(improvement.mean(), color="#f39c12", linewidth=2,
                linestyle="-", label=f"Mean Δ={improvement.mean():.3f}")
    ax5.set_title("Hybrid Improvement over\nBest Single Model (per district)", fontsize=12)
    ax5.set_xlabel("Δ Spearman ρ (Hybrid − Best Single)")
    ax5.set_ylabel("Number of districts")
    ax5.legend(fontsize=9)
    ax5.grid(axis="y")
    textstr = f"Hybrid better: {wins} districts\nHybrid worse:  {losses} districts\nTied: {ties}"
    ax5.text(0.97, 0.95, textstr, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='#2a2f45', alpha=0.8))

    out_path = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\n  Chart saved → {out_path}")
    return out_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  Model Comparison — Historical Evaluation")
    print("="*60)

    print("\n[1/3] Loading and aggregating data ...")
    data, learned = load_data()

    print("\n[2/3] Adding suitability scores ...")
    data = add_suitability(data)
    data = data.dropna(subset=["suitability_score"])

    print("\n[3/3] Evaluating models across districts ...")
    df_results, top1_acc, n_districts = evaluate(data, learned)

    print("\nMean Spearman correlations:")
    for m in MODEL_COLORS:
        print(f"  {m:<22}: rho={df_results[m].mean():.4f}  top1={top1_acc[m]:.1f}%")

    plot(df_results, top1_acc, n_districts, learned)
    print("\nDone.")


if __name__ == "__main__":
    main()
