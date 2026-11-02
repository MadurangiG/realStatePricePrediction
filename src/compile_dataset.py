"""
Melbourne Eastern Suburbs Housing Dataset Compiler
Merges live scraped property listings with historical market transactions for
Burwood East and surrounding eastern suburbs. Enriches all records with
geodesic proximity to local schools (MWSC, GWSC, Deakin), retail hubs
(Burwood Brickworks, Burwood One, The Glen, Box Hill Central), public transport
(Route 75 Tram, train stations), and Melbourne CBD.
"""

import os
import io
import math
import requests
import pandas as pd
import numpy as np
from enrich_poi import enrich_property_with_pois, SUBURB_CENTROIDS

# Target Eastern Suburbs centered around Burwood East
EASTERN_SUBURBS = [
    "Burwood East", "Burwood", "Mount Waverley", "Glen Waverley",
    "Box Hill", "Box Hill South", "Box Hill North",
    "Blackburn", "Blackburn South", "Blackburn North",
    "Forest Hill", "Vermont", "Vermont South",
    "Ashwood", "Chadstone", "Surrey Hills", "Camberwell"
]

TARGET_SUBURB_LOWER = set(s.lower() for s in EASTERN_SUBURBS)

HISTORICAL_DATA_URL = "https://raw.githubusercontent.com/dipalira/Melbourne-Housing-Data-Kaggle/master/Data/Melbourne_housing_FULL.csv"


def load_and_filter_historical_sales() -> pd.DataFrame:
    """
    Downloads and extracts property sales for the target Eastern Suburbs.
    """
    print(f"Fetching historical sales data from {HISTORICAL_DATA_URL}...")
    resp = requests.get(HISTORICAL_DATA_URL, timeout=30)
    resp.raise_for_status()
    
    df_raw = pd.read_csv(io.BytesIO(resp.content))
    print(f"Total raw sales records downloaded: {len(df_raw)}")
    
    # Filter for target suburbs
    df_raw["Suburb_clean"] = df_raw["Suburb"].astype(str).str.strip()
    mask = df_raw["Suburb_clean"].str.lower().isin(TARGET_SUBURB_LOWER)
    df_filtered = df_raw[mask].copy()
    print(f"Filtered records in Burwood East & surrounding suburbs: {len(df_filtered)}")
    
    # Keep records with valid prices
    df_filtered = df_filtered[df_filtered["Price"].notnull() & (df_filtered["Price"] >= 200_000)].copy()
    print(f"Records with valid sales prices: {len(df_filtered)}")
    
    # Standardize column names
    # Mapping property types: h=House, u=Unit/Apartment, t=Townhouse
    type_map = {"h": "House", "u": "Unit", "t": "Townhouse"}
    df_filtered["property_type"] = df_filtered["Type"].map(type_map).fillna("House")
    
    # Harmonize structural attributes
    df_filtered["bedrooms"] = df_filtered["Bedroom2"].combine_first(df_filtered["Rooms"])
    df_filtered["bathrooms"] = df_filtered["Bathroom"]
    df_filtered["parking_spaces"] = df_filtered["Car"]
    df_filtered["land_size_sqm"] = df_filtered["Landsize"]
    df_filtered["building_size_sqm"] = df_filtered["BuildingArea"]
    df_filtered["year_built"] = df_filtered["YearBuilt"]
    df_filtered["price_aud"] = df_filtered["Price"]
    df_filtered["suburb"] = df_filtered["Suburb_clean"]
    df_filtered["postcode"] = df_filtered["Postcode"].fillna(3151).astype(int)
    df_filtered["address"] = df_filtered["Address"] + ", " + df_filtered["suburb"] + " VIC " + df_filtered["postcode"].astype(str)
    df_filtered["latitude"] = df_filtered["Lattitude"]
    df_filtered["longitude"] = df_filtered["Longtitude"]
    df_filtered["source"] = "Historical_Sales_VIC"
    
    # Impute missing coordinates using suburb centroid
    for idx, row in df_filtered.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        sub = str(row["suburb"]).lower()
        if pd.isna(lat) or pd.isna(lon) or lat == 0:
            if sub in SUBURB_CENTROIDS:
                df_filtered.at[idx, "latitude"] = SUBURB_CENTROIDS[sub][0] + np.random.normal(0, 0.003)
                df_filtered.at[idx, "longitude"] = SUBURB_CENTROIDS[sub][1] + np.random.normal(0, 0.003)
            else:
                df_filtered.at[idx, "latitude"] = SUBURB_CENTROIDS["burwood east"][0]
                df_filtered.at[idx, "longitude"] = SUBURB_CENTROIDS["burwood east"][1]

    keep_cols = [
        "address", "suburb", "postcode", "property_type",
        "bedrooms", "bathrooms", "parking_spaces",
        "land_size_sqm", "building_size_sqm", "year_built",
        "latitude", "longitude", "price_aud", "source"
    ]
    return df_filtered[keep_cols]


def load_live_scraped_data(scraped_csv: str = "data/raw/melbourne_housing_raw.csv") -> pd.DataFrame:
    """
    Loads properties scraped live from web listing portal.
    """
    if not os.path.exists(scraped_csv):
        print(f"Warning: {scraped_csv} not found, proceeding with historical compilation only.")
        return pd.DataFrame()
        
    df_scraped = pd.read_csv(scraped_csv)
    # Filter for valid price
    df_scraped = df_scraped[df_scraped["price_aud"].notnull() & (df_scraped["price_aud"] >= 200_000)].copy()
    df_scraped["source"] = "Live_Scraped_2026"
    df_scraped["building_size_sqm"] = np.nan
    df_scraped["year_built"] = np.nan
    
    keep_cols = [
        "address", "suburb", "postcode", "property_type",
        "bedrooms", "bathrooms", "parking_spaces",
        "land_size_sqm", "building_size_sqm", "year_built",
        "latitude", "longitude", "price_aud", "source"
    ]
    # Ensure all columns exist
    for c in keep_cols:
        if c not in df_scraped.columns:
            df_scraped[c] = np.nan
            
    return df_scraped[keep_cols]


def build_final_dataset(output_path: str = "data/processed/melbourne_burwood_east_housing.csv") -> pd.DataFrame:
    """
    Assembles the final master dataset, enriches with all spatial POI features,
    and performs data validation.
    """
    print("\n" + "=" * 70)
    print("ASSEMBLING MASTER DATASET FOR BURWOOD EAST & SURROUNDS")
    print("=" * 70)
    
    # 1. Load historical sales and live scraped data
    df_hist = load_and_filter_historical_sales()
    df_live = load_live_scraped_data()
    
    print(f"Combining {len(df_hist)} historical records + {len(df_live)} live scraped records...")
    df_combined = pd.concat([df_hist, df_live], ignore_index=True)
    
    # Remove duplicate addresses
    df_combined = df_combined.drop_duplicates(subset=["address", "suburb", "price_aud"]).copy()
    print(f"Unique properties after deduplication: {len(df_combined)}")
    
    # 2. Enrich all rows with POI spatial metrics
    print("Computing geodesic distances to local schools, shops, stations, and CBD...")
    poi_rows = []
    for _, row in df_combined.iterrows():
        poi_data = enrich_property_with_pois(
            lat=row["latitude"],
            lon=row["longitude"],
            suburb=str(row["suburb"])
        )
        poi_rows.append(poi_data)
        
    df_poi = pd.DataFrame(poi_rows)
    
    # Merge POI features back
    for col in df_poi.columns:
        if col not in ["latitude", "longitude"]:
            df_combined[col] = df_poi[col]
            
    # Update latitude/longitude with validated coordinates
    df_combined["latitude"] = df_poi["latitude"]
    df_combined["longitude"] = df_poi["longitude"]
    
    # 3. Clean & Impute structural features
    # Impute bedrooms: median per property type
    df_combined["bedrooms"] = df_combined.groupby("property_type")["bedrooms"].transform(
        lambda s: s.fillna(s.median())
    ).fillna(3).astype(int)
    
    # Impute bathrooms: median per property type and bedrooms
    df_combined["bathrooms"] = df_combined.groupby(["property_type", "bedrooms"])["bathrooms"].transform(
        lambda s: s.fillna(s.median())
    ).fillna(2).astype(int)
    
    # Impute parking: median per property type
    df_combined["parking_spaces"] = df_combined.groupby("property_type")["parking_spaces"].transform(
        lambda s: s.fillna(s.median())
    ).fillna(1).astype(int)
    
    # Land size: replace 0 or NA with suburb median for the property type
    df_combined.loc[df_combined["land_size_sqm"] <= 10, "land_size_sqm"] = np.nan
    df_combined["land_size_sqm"] = df_combined.groupby(["suburb", "property_type"])["land_size_sqm"].transform(
        lambda s: s.fillna(s.median())
    )
    # Global fallback for land size
    df_combined["land_size_sqm"] = df_combined["land_size_sqm"].fillna(df_combined["land_size_sqm"].median()).round(1)
    
    # Add price per square meter
    df_combined["price_per_sqm"] = (df_combined["price_aud"] / df_combined["land_size_sqm"]).round(2)
    
    # Cap extreme anomalies (e.g. land size > 3000 sqm or price > 6,000,000)
    df_combined = df_combined[
        (df_combined["price_aud"] >= 250_000) &
        (df_combined["price_aud"] <= 5_000_000) &
        (df_combined["land_size_sqm"] >= 50) &
        (df_combined["land_size_sqm"] <= 2500)
    ].copy()
    
    # Save processed master dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"\nFinal dataset compiled successfully: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    print(f"Saved to: {output_path}")
    
    print("\nSuburb Breakdown:")
    print(df_combined["suburb"].value_counts())
    
    print("\nSummary Statistics of Target (price_aud):")
    print(df_combined["price_aud"].describe().apply(lambda x: f"${x:,.0f}"))
    
    return df_combined


if __name__ == "__main__":
    df = build_final_dataset()
