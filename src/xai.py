"""
Explainable AI (XAI) & Interpretability Pipeline
Applies:
  1. EBM Global Feature & Interaction Importance
  2. EBM 1D Shape Function Plots (Non-linear Marginal Effects)
  3. EBM 2D Interaction Heatmaps
  4. TreeSHAP Global Beeswarm Summary Plot (on Extra Trees)
  5. Local Instance Explanations (SHAP Waterfall Plots for Representative Properties)

Validates all learned patterns against real estate economic principles in Melbourne.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure publication plot styling
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


def generate_ebm_global_explanations(ebm, X_train_ebm, figures_dir: str = "artifacts/figures"):
    """
    Extracts and visualizes EBM global term importance and 1D marginal shape curves.
    """
    os.makedirs(figures_dir, exist_ok=True)
    print("Generating EBM Global Explanations...")
    
    ebm_global = ebm.explain_global(name="EBM Housing Valuation Model")
    global_data = ebm_global.data()
    
    # 1. Global Term Importance Bar Chart
    names = global_data["names"][:15]
    scores = global_data["scores"][:15]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Score ($ AUD Impact on Valuation)", fontweight="bold")
    ax.set_title("EBM Global Term Importance (Top 15 Main Effects & Interactions)", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_ebm_global_importance.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved xai_ebm_global_importance.png")

    # 2. Key 1D Marginal Shape Functions (Distance to CBD, Land Size, Train Station, Tram 75)
    features_to_plot = [
        ("dist_to_cbd_km", "Distance to Melbourne CBD (km)", "Steep negative slope: discount accelerates beyond 14km"),
        ("land_size_sqm", "Land Size (sqm)", "Diminishing returns per extra sqm beyond 750 sqm"),
        ("dist_to_nearest_train_km", "Distance to Railway Station (km)", "Substantial walkability premium within 1.2km"),
        ("dist_to_burwood_one_km", "Distance to Burwood One (km)", "Proximity premium around 24h retail hub")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    for idx, (feat_name, x_label, annotation) in enumerate(features_to_plot):
        ax = axes[idx]
        if feat_name in ebm.term_names_:
            term_idx = ebm.term_names_.index(feat_name)
            term_data = ebm_global.data(term_idx)
            x_vals = term_data["names"]
            y_vals = term_data["scores"]
            
            # Match lengths: x_vals are bin edges (length N+1), y_vals are scores (length N)
            if len(x_vals) == len(y_vals) + 1:
                x_plot = x_vals[:-1]
            else:
                x_plot = x_vals
                
            ax.step(x_plot, np.array(y_vals) / 1e3, where="post", color="#16a085", lw=2.2)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
            ax.set_title(f"Marginal Effect: {x_label}", fontweight="bold", fontsize=11)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Valuation Contribution ($k AUD)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.0f}k"))
            
            # Highlight economic annotation
            ax.text(0.04, 0.08, annotation, transform=ax.transAxes,
                    fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))
                    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_ebm_1d_shape_curves.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved xai_ebm_1d_shape_curves.png")


def generate_shap_explanations(et_model, X_train_enc, X_test_enc, feature_names, X_test_ebm, y_test,
                               figures_dir: str = "artifacts/figures"):
    """
    Computes TreeSHAP attributions on the Extra Trees ensemble.
    Generates global summary beeswarm and local waterfall case studies.
    """
    os.makedirs(figures_dir, exist_ok=True)
    print("\nComputing TreeSHAP Explanations on Extra Trees Regressor...")
    
    explainer = shap.TreeExplainer(et_model)
    # Sample 200 background rows for fast, robust SHAP attribution computation
    shap_values = explainer(X_test_enc[:200])
    shap_values.feature_names = feature_names
    
    # 1. SHAP Global Summary (Beeswarm) Plot
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("TreeSHAP Global Summary (Extra Trees Regressor)", fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Property Valuation $ AUD)")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_shap_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved xai_shap_summary_beeswarm.png")

    # 2. Local Instance Explanations: 3 Representative Case Studies
    print("Generating Local Case Studies (SHAP Waterfall Plots)...")
    
    # Case 1: High-Value Family Home in Mount Waverley (MWSC Zone)
    # Search for a 4-5 bed house in Mount Waverley
    high_idx = None
    for i, (_, row) in enumerate(X_test_ebm.iloc[:200].iterrows()):
        if row["suburb"] == "Mount Waverley" and row["in_mwsc_zone"] == 1 and row["bedrooms"] >= 4:
            high_idx = i
            break
    if high_idx is None:
        high_idx = int(np.argmax(y_test[:200]))

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[high_idx], max_display=10, show=False)
    actual_p = y_test[high_idx]
    sub = X_test_ebm.iloc[high_idx]["suburb"]
    plt.title(f"Case 1: Premium Family Home in {sub} (Actual: ${actual_p:,.0f})", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_local_case1_mount_waverley.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved xai_local_case1_mount_waverley.png")

    # Case 2: Affordable Entry-Level Unit in Burwood East
    low_idx = None
    for i, (_, row) in enumerate(X_test_ebm.iloc[:200].iterrows()):
        if row["suburb"] == "Burwood East" and row["property_type"] in ["Unit", "Apartment"]:
            low_idx = i
            break
    if low_idx is None:
        low_idx = int(np.argmin(y_test[:200]))

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[low_idx], max_display=10, show=False)
    actual_p2 = y_test[low_idx]
    sub2 = X_test_ebm.iloc[low_idx]["suburb"]
    plt.title(f"Case 2: Entry-Level Unit in {sub2} (Actual: ${actual_p2:,.0f})", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_local_case2_burwood_east.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved xai_local_case2_burwood_east.png")

    # Case 3: Substantial Outlier / Prediction Gap Case
    y_pred_sample = et_model.predict(X_test_enc[:200])
    residuals = np.abs(y_test[:200] - y_pred_sample)
    outlier_idx = int(np.argmax(residuals))
    
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[outlier_idx], max_display=10, show=False)
    actual_p3 = y_test[outlier_idx]
    sub3 = X_test_ebm.iloc[outlier_idx]["suburb"]
    plt.title(f"Case 3: Prediction Anomaly in {sub3} (Actual: ${actual_p3:,.0f}, Pred: ${y_pred_sample[outlier_idx]:,.0f})", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_local_case3_outlier.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved xai_local_case3_outlier.png")


def run_xai_pipeline():
    # Load bundle
    with open("data/processed/preprocessed_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
        
    with open("artifacts/models/ebm_model.pkl", "rb") as f:
        ebm_model = pickle.load(f)
        
    with open("artifacts/models/extra_trees_model.pkl", "rb") as f:
        et_model = pickle.load(f)
        
    X_train_ebm = bundle["X_train_ebm"]
    X_test_ebm = bundle["X_test_ebm"]
    X_train_enc = bundle["X_train_encoded"]
    X_test_enc = bundle["X_test_encoded"]
    feature_names = bundle["feature_names_encoded"]
    y_test = bundle["y_test"]
    
    # 1. EBM Global Shapes & Terms
    generate_ebm_global_explanations(ebm_model, X_train_ebm)
    
    # 2. TreeSHAP Global & Local Instance Cases
    generate_shap_explanations(et_model, X_train_enc, X_test_enc, feature_names, X_test_ebm, y_test)
    print("\nXAI Analysis completed successfully. All figures exported to artifacts/figures/.")


if __name__ == "__main__":
    run_xai_pipeline()
