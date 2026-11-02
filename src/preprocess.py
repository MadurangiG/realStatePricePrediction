"""
Data Preprocessing & Exploratory Data Analysis (EDA) Pipeline
Performs feature engineering, statistical transformations, train/test splitting,
and generates publication-quality EDA figures for the academic report.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set aesthetic style for academic paper figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "artifacts/figures"):
    """
    Generates and saves core exploratory figures for the written report.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating publication-quality EDA figures...")
    
    # 1. Target Price Distribution (Raw & Log-transformed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.histplot(df["price_aud"] / 1e6, kde=True, ax=axes[0], color="#1f77b4", bins=30)
    axes[0].set_title("Property Price Distribution (Raw)", fontweight="bold")
    axes[0].set_xlabel("Price ($ Millions AUD)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["price_aud"].median() / 1e6, color="red", linestyle="--", label=f"Median: ${df['price_aud'].median()/1e6:.2f}M")
    axes[0].legend()
    
    sns.histplot(np.log10(df["price_aud"]), kde=True, ax=axes[1], color="#2ca02c", bins=30)
    axes[1].set_title("Property Price Distribution (Log10)", fontweight="bold")
    axes[1].set_xlabel("Log10(Price AUD)")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_price_distribution.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved eda_price_distribution.png")

    # 2. Suburb Median Price Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    suburb_order = df.groupby("suburb")["price_aud"].median().sort_values(ascending=False).index
    
    sns.barplot(data=df, x="price_aud", y="suburb", order=suburb_order,
                estimator=np.median, errorbar=None, palette="viridis", ax=ax)
    ax.set_title("Median Property Price by Suburb (Burwood East & Surrounds)", fontweight="bold")
    ax.set_xlabel("Median Price ($ AUD)")
    ax.set_ylabel("Suburb")
    # Format x-axis with dollar formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.2f}M"))
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_suburb_median_prices.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved eda_suburb_median_prices.png")

    # 3. Spatial Amenity / School Zone Price Premium (MWSC & GWSC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mount Waverley Secondary College Zone
    sns.boxplot(data=df[df["dist_to_mwsc_km"] <= 4.0], x="in_mwsc_zone", y="price_aud",
                palette=["#e74c3c", "#2ecc71"], ax=axes[0])
    axes[0].set_xticklabels(["Outside Zone", "Inside MWSC Zone"])
    axes[0].set_title("Mount Waverley Secondary Zone Premium", fontweight="bold")
    axes[0].set_ylabel("Price ($ AUD)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.2f}M"))
    
    # Glen Waverley Secondary College Zone
    sns.boxplot(data=df[df["dist_to_gwsc_km"] <= 4.5], x="in_gwsc_zone", y="price_aud",
                palette=["#e74c3c", "#2ecc71"], ax=axes[1])
    axes[1].set_xticklabels(["Outside Zone", "Inside GWSC Zone"])
    axes[1].set_title("Glen Waverley Secondary Zone Premium", fontweight="bold")
    axes[1].set_ylabel("Price ($ AUD)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.2f}M"))
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_school_zone_premiums.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved eda_school_zone_premiums.png")

    # 4. Correlation Heatmap of Structural & POI Features
    corr_cols = [
        "price_aud", "bedrooms", "bathrooms", "parking_spaces", "land_size_sqm",
        "dist_to_mwsc_km", "dist_to_gwsc_km", "dist_to_deakin_km",
        "dist_to_burwood_one_km", "dist_to_the_glen_km", "dist_to_nearest_train_km",
        "dist_to_tram_75_km", "dist_to_cbd_km"
    ]
    corr_df = df[corr_cols].rename(columns={
        "price_aud": "Price",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "parking_spaces": "Cars",
        "land_size_sqm": "Land (sqm)",
        "dist_to_mwsc_km": "Dist MWSC",
        "dist_to_gwsc_km": "Dist GWSC",
        "dist_to_deakin_km": "Dist Deakin",
        "dist_to_burwood_one_km": "Dist Burwood One",
        "dist_to_the_glen_km": "Dist The Glen",
        "dist_to_nearest_train_km": "Dist Train",
        "dist_to_tram_75_km": "Dist Tram 75",
        "dist_to_cbd_km": "Dist CBD"
    }).corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"label": "Pearson Correlation"}, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix: Property Price vs. Structural & POI Amenity Drivers", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved eda_correlation_heatmap.png")


def prepare_dataset_splits(data_path: str = "data/processed/melbourne_burwood_east_housing.csv",
                           output_dir: str = "data/processed",
                           test_size: float = 0.20,
                           random_state: int = 42):
    """
    Cleans, encodes, and splits dataset into train and test partitions.
    Saves preprocessed arrays and feature transformer metadata.
    """
    df = pd.read_csv(data_path)
    generate_eda_figures(df)
    
    # Define features
    numerical_features = [
        "bedrooms", "bathrooms", "parking_spaces", "land_size_sqm",
        "dist_to_mwsc_km", "dist_to_gwsc_km", "dist_to_box_hill_high_km",
        "dist_to_deakin_km", "dist_to_burwood_one_km", "dist_to_burwood_brickworks_km",
        "dist_to_the_glen_km", "dist_to_box_hill_central_km", "dist_to_nearest_major_shop_km",
        "dist_to_nearest_train_km", "dist_to_tram_75_km", "dist_to_cbd_km",
        "latitude", "longitude"
    ]
    
    binary_features = [
        "in_mwsc_zone", "in_gwsc_zone", "in_box_hill_high_zone"
    ]
    
    categorical_features = [
        "suburb", "property_type"
    ]
    
    target_col = "price_aud"
    
    feature_cols = numerical_features + binary_features + categorical_features
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 80/20 Stratified train-test split by price quartile to preserve price distribution
    price_quartiles = pd.qcut(y, q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=price_quartiles
    )
    
    print(f"\nTrain size: {len(X_train)} ({len(X_train)/len(X):.0%}) | Test size: {len(X_test)} ({len(X_test)/len(X):.0%})")
    
    # Save raw split dataframes for EBM (EBM handles categoricals natively without one-hot encoding!)
    train_df = X_train.copy()
    train_df["price_aud"] = y_train
    test_df = X_test.copy()
    test_df["price_aud"] = y_test
    
    train_df.to_csv(os.path.join(output_dir, "train_raw.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_raw.csv"), index=False)
    
    # Also create One-Hot Encoded & Scaled matrices for standard models (Ridge, Random Forest, Extra Trees)
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("bin", bin_pipeline, binary_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Retrieve clean feature names post-one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numerical_features + binary_features + encoded_cat_names
    
    # Save transformed datasets and preprocessor pipeline
    processed_bundle = {
        "X_train_ebm": X_train,
        "X_test_ebm": X_test,
        "X_train_encoded": X_train_transformed,
        "X_test_encoded": X_test_transformed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names_encoded": all_feature_names,
        "numerical_features": numerical_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor
    }
    
    bundle_path = os.path.join(output_dir, "preprocessed_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(processed_bundle, f)
        
    print(f"Preprocessed bundle saved successfully to {bundle_path}")
    print(f"Total encoded features for standard trees/linear models: {len(all_feature_names)}")
    return processed_bundle


if __name__ == "__main__":
    prepare_dataset_splits()
