"""
merge_yield_data.py
-------------------
Merges yield data from kerala_monthly_estimates_2023_2025.csv into each
crop price CSV by matching on Year + Month.

For months/crops where yield is NaN in the source CSV:
  - First tries forward-fill then back-fill within the same crop
  - If still NaN (crop not in yield CSV at all), estimates from known
    agronomic averages for Kerala

Run from the project root:
    python src/merge_yield_data.py
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Map CSV filename -> crop name in the yield dataset (None = not present)
CROP_FILE_MAP = {
    "banana_data.csv":  None,      # Not in yield CSV → use agronomic estimate
    "cocunut_data.csv": "Coconut",
    "cashew_data.csv":  "Cashew",
    "coffee_data.csv":  "Coffee",
    "rice_data.csv":    "Paddy",   # Paddy covers Rice in yield CSV
    "tapioca_data.csv": None,      # Not in yield CSV → use agronomic estimate
    "wheat_data.csv":   None,      # Not in yield CSV → use agronomic estimate
}

# Agronomic fallback yield in kg/ha for crops absent from the yield CSV
# Sources: Kerala State Planning Board averages
FALLBACK_YIELD_KG_HA = {
    "banana_data.csv":  24710.0,   # ~100 q/acre × 247.105
    "tapioca_data.csv": 29652.6,   # ~120 q/acre × 247.105
    "wheat_data.csv":    3706.6,   # ~15 q/acre × 247.105
}

YIELD_CSV = os.path.join(DATA_DIR, "kerala_monthly_estimates_2023_2025.csv")


def build_full_yield_lookup():
    """
    Build a complete (Crop, Year, Month) → yield_kg_per_ha lookup table.
    Fills NaN months within a crop by ffill/bfill over the full 2023-2025 range.
    """
    df = pd.read_csv(YIELD_CSV)
    df["Crop"] = df["Crop"].str.strip().str.title()

    # Use Paddy as Rice too
    paddy_rows = df[df["Crop"] == "Paddy"].copy()
    paddy_rows["Crop"] = "Rice"
    df = pd.concat([df, paddy_rows], ignore_index=True)

    yield_col = "Estimated_Productivity_kg_per_ha"

    # For each crop, pivot to a full Year×Month grid, fill gaps, then melt back
    all_crops = []
    for crop, grp in df.groupby("Crop"):
        # Create all 36 (year, month) combos for 2023-2025
        idx = pd.MultiIndex.from_product(
            [[2023, 2024, 2025], range(1, 13)], names=["Year", "Month"]
        )
        # Deduplicate on (Year, Month) before setting index to avoid non-unique error
        grp_dedup = grp.drop_duplicates(subset=["Year", "Month"])
        full = grp_dedup.set_index(["Year", "Month"])[[yield_col]].reindex(idx)

        # For each year the crop has a known annual yield, broadcast to all months
        yearly = grp.groupby("Year")[yield_col].first()
        for yr, val in yearly.items():
            if pd.notna(val):
                mask = full.index.get_level_values("Year") == yr
                full.loc[mask, yield_col] = full.loc[mask, yield_col].fillna(val)

        # Any remaining NaN → ffill then bfill across months
        full[yield_col] = full[yield_col].ffill().bfill()

        full = full.reset_index()
        full["Crop"] = crop
        all_crops.append(full)

    lookup = pd.concat(all_crops, ignore_index=True)
    lookup = lookup.rename(columns={yield_col: "yield_kg_per_ha"})
    return lookup[["Crop", "Year", "Month", "yield_kg_per_ha"]]


def merge_into_price_csv(crop_file, yield_crop_name, lookup_df):
    path = os.path.join(DATA_DIR, crop_file)
    if not os.path.exists(path):
        print(f"  SKIP (file not found): {crop_file}")
        return

    price_df = pd.read_csv(path)

    if "t" not in price_df.columns:
        print(f"  SKIP (no 't' column): {crop_file}")
        return

    # Extract year + month from date column
    price_df["_date"]  = pd.to_datetime(price_df["t"], errors="coerce")
    price_df["_year"]  = price_df["_date"].dt.year.astype("Int64")
    price_df["_month"] = price_df["_date"].dt.month.astype("Int64")

    # Drop old yield column if re-running
    if "yield_kg_per_ha" in price_df.columns:
        price_df = price_df.drop(columns=["yield_kg_per_ha"])

    if yield_crop_name is not None:
        # Join from the lookup table on Year + Month
        crop_yield = lookup_df[lookup_df["Crop"] == yield_crop_name][
            ["Year", "Month", "yield_kg_per_ha"]
        ].copy()
        crop_yield = crop_yield.rename(columns={"Year": "_year", "Month": "_month"})
        crop_yield["_year"]  = crop_yield["_year"].astype("Int64")
        crop_yield["_month"] = crop_yield["_month"].astype("Int64")

        merged = price_df.merge(crop_yield, on=["_year", "_month"], how="left")
        # Any remaining NaN (year outside 2023-2025) → fill with overall crop mean
        overall_mean = crop_yield["yield_kg_per_ha"].mean()
        merged["yield_kg_per_ha"] = merged["yield_kg_per_ha"].fillna(overall_mean)
    else:
        # Crop not in yield CSV → use agronomic fallback constant
        merged = price_df.copy()
        fallback = FALLBACK_YIELD_KG_HA.get(crop_file, np.nan)
        merged["yield_kg_per_ha"] = fallback
        print(f"  ESTIMATE: {crop_file} — using fallback {fallback:.1f} kg/ha for all rows")

    # Clean up helper columns
    merged = merged.drop(columns=["_date", "_year", "_month"])
    merged.to_csv(path, index=False)

    matched = merged["yield_kg_per_ha"].notna().sum()
    total   = len(merged)
    sample  = merged["yield_kg_per_ha"].dropna().iloc[0] if matched > 0 else "N/A"
    if yield_crop_name is not None:
        print(f"  OK: {crop_file} — {matched}/{total} rows | sample yield={sample:.2f} kg/ha")


def main():
    print("Building full yield lookup (Year + Month) with gap-filling...")
    lookup = build_full_yield_lookup()
    print(f"  Crops: {sorted(lookup['Crop'].unique())}")
    print(f"  Years: {sorted(lookup['Year'].unique())}\n")

    print("Merging into price CSVs...")
    for csv_file, crop_name in CROP_FILE_MAP.items():
        merge_into_price_csv(csv_file, crop_name, lookup)

    print("\nDone. Every price CSV now has a 'yield_kg_per_ha' column.")


if __name__ == "__main__":
    main()
