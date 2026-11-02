"""
Model Training, Optimization & Benchmarking Pipeline
Trains, tunes, and evaluates:
  1. Explainable Boosting Machine (EBM / GA2M) [New Algorithm 1]
  2. Extra Trees Regressor (Extremely Randomized Trees) [New Algorithm 2]
  3. Random Forest Regressor [Covered Baseline 1]
  4. Ridge Regression [Covered Baseline 2]

Produces evaluation metrics tables (R2, RMSE, MAE, MAPE) and diagnostic plots.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from interpret.glassbox import ExplainableBoostingRegressor

# Set styling for publication plots
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


def load_preprocessed_bundle(bundle_path: str = "data/processed/preprocessed_bundle.pkl") -> dict:
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes R2, RMSE ($AUD), MAE ($AUD), and MAPE (%).
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "R2": round(r2, 4),
        "RMSE ($)": round(rmse, 2),
        "MAE ($)": round(mae, 2),
        "MAPE (%)": round(mape, 2)
    }


def train_and_benchmark_models(bundle_path: str = "data/processed/preprocessed_bundle.pkl",
                               models_dir: str = "artifacts/models",
                               figures_dir: str = "artifacts/figures") -> pd.DataFrame:
    """
    Executes training and evaluation across all 4 models using 5-fold CV and test holdout.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    bundle = load_preprocessed_bundle(bundle_path)
    X_train_ebm = bundle["X_train_ebm"]
    X_test_ebm = bundle["X_test_ebm"]
    X_train_enc = bundle["X_train_encoded"]
    X_test_enc = bundle["X_test_encoded"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # -------------------------------------------------------------
    # Model 1: Ridge Regression (Baseline 1)
    # -------------------------------------------------------------
    print("\n[1/4] Training Baseline Model: Ridge Regression...")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge_cv = cross_val_score(ridge, X_train_enc, y_train, cv=kf, scoring="r2")
    ridge.fit(X_train_enc, y_train)
    y_pred_ridge = ridge.predict(X_test_enc)
    metrics_ridge = compute_metrics(y_test, y_pred_ridge)
    metrics_ridge["CV_R2_Mean"] = round(ridge_cv.mean(), 4)
    metrics_ridge["CV_R2_Std"] = round(ridge_cv.std(), 4)
    print(f"  Ridge Test R2: {metrics_ridge['R2']:.4f} | RMSE: ${metrics_ridge['RMSE ($)']:,.0f} | MAE: ${metrics_ridge['MAE ($)']:,.0f}")
    
    # -------------------------------------------------------------
    # Model 2: Random Forest Regressor (Baseline 2)
    # -------------------------------------------------------------
    print("\n[2/4] Training Baseline Model: Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf_cv = cross_val_score(rf, X_train_enc, y_train, cv=kf, scoring="r2")
    rf.fit(X_train_enc, y_train)
    y_pred_rf = rf.predict(X_test_enc)
    metrics_rf = compute_metrics(y_test, y_pred_rf)
    metrics_rf["CV_R2_Mean"] = round(rf_cv.mean(), 4)
    metrics_rf["CV_R2_Std"] = round(rf_cv.std(), 4)
    print(f"  Random Forest Test R2: {metrics_rf['R2']:.4f} | RMSE: ${metrics_rf['RMSE ($)']:,.0f} | MAE: ${metrics_rf['MAE ($)']:,.0f}")

    # -------------------------------------------------------------
    # Model 3: Extra Trees Regressor (New Algorithm 1)
    # -------------------------------------------------------------
    print("\n[3/4] Training New Model: Extra Trees Regressor (Extremely Randomized Trees)...")
    et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    et_cv = cross_val_score(et, X_train_enc, y_train, cv=kf, scoring="r2")
    et.fit(X_train_enc, y_train)
    y_pred_et = et.predict(X_test_enc)
    metrics_et = compute_metrics(y_test, y_pred_et)
    metrics_et["CV_R2_Mean"] = round(et_cv.mean(), 4)
    metrics_et["CV_R2_Std"] = round(et_cv.std(), 4)
    print(f"  Extra Trees Test R2: {metrics_et['R2']:.4f} | RMSE: ${metrics_et['RMSE ($)']:,.0f} | MAE: ${metrics_et['MAE ($)']:,.0f}")

    # -------------------------------------------------------------
    # Model 4: Explainable Boosting Machine (EBM) (New Algorithm 2 - Primary Glass-Box)
    # -------------------------------------------------------------
    print("\n[4/4] Training New Model: Explainable Boosting Machine (EBM / GA2M)...")
    ebm = ExplainableBoostingRegressor(
        interactions=15,
        max_bins=256,
        outer_bags=12,
        inner_bags=0,
        learning_rate=0.04,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    ebm.fit(X_train_ebm, y_train)
    y_pred_ebm = ebm.predict(X_test_ebm)
    metrics_ebm = compute_metrics(y_test, y_pred_ebm)
    
    # 5-fold CV for EBM
    ebm_cv_scores = []
    for train_idx, val_idx in kf.split(X_train_ebm):
        X_tr, X_val = X_train_ebm.iloc[train_idx], X_train_ebm.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        fold_ebm = ExplainableBoostingRegressor(
            interactions=10, max_bins=128, outer_bags=8, learning_rate=0.04,
            random_state=42, n_jobs=-1
        )
        fold_ebm.fit(X_tr, y_tr)
        val_pred = fold_ebm.predict(X_val)
        ebm_cv_scores.append(r2_score(y_val, val_pred))
    
    metrics_ebm["CV_R2_Mean"] = round(np.mean(ebm_cv_scores), 4)
    metrics_ebm["CV_R2_Std"] = round(np.std(ebm_cv_scores), 4)
    print(f"  EBM Test R2: {metrics_ebm['R2']:.4f} | RMSE: ${metrics_ebm['RMSE ($)']:,.0f} | MAE: ${metrics_ebm['MAE ($)']:,.0f}")

    # -------------------------------------------------------------
    # Compile Comparison Table
    # -------------------------------------------------------------
    results = [
        {"Model": "Ridge Regression", "Category": "Baseline (Covered)", **metrics_ridge},
        {"Model": "Random Forest Regressor", "Category": "Baseline (Covered)", **metrics_rf},
        {"Model": "Extra Trees Regressor", "Category": "New Algorithm (Ensemble)", **metrics_et},
        {"Model": "Explainable Boosting Machine (EBM)", "Category": "New Algorithm (Glass-Box GA2M)", **metrics_ebm}
    ]
    df_results = pd.DataFrame(results)
    
    # Save results table
    table_csv_path = "artifacts/model_benchmark_results.csv"
    df_results.to_csv(table_csv_path, index=False)
    print("\n" + "=" * 80)
    print("FINAL MODEL BENCHMARK COMPARISON TABLE (BURWOOD EAST & SURROUNDS)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # Save model weights / objects
    with open(os.path.join(models_dir, "ridge_model.pkl"), "wb") as f:
        pickle.dump(ridge, f)
    with open(os.path.join(models_dir, "random_forest_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(models_dir, "extra_trees_model.pkl"), "wb") as f:
        pickle.dump(et, f)
    with open(os.path.join(models_dir, "ebm_model.pkl"), "wb") as f:
        pickle.dump(ebm, f)
    print(f"\nAll 4 trained models serialized to {models_dir}/")

    # -------------------------------------------------------------
    # Generate Diagnostic Comparison Plots
    # -------------------------------------------------------------
    # 1. Bar Chart of Test R2 & RMSE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = ["#95a5a6", "#7f8c8d", "#3498db", "#2ecc71"]
    
    sns.barplot(data=df_results, x="Model", y="R2", palette=palette, ax=axes[0])
    axes[0].set_title("Test R² Score Comparison (Higher is Better)", fontweight="bold")
    axes[0].set_ylim(0.5, 0.90)
    axes[0].set_ylabel("R² Score")
    axes[0].set_xticklabels(df_results["Model"], rotation=15, ha="right")
    for i, v in enumerate(df_results["R2"]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    sns.barplot(data=df_results, x="Model", y="MAE ($)", palette=palette, ax=axes[1])
    axes[1].set_title("Test Mean Absolute Error (MAE) ($AUD, Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("MAE ($ AUD)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    axes[1].set_xticklabels(df_results["Model"], rotation=15, ha="right")
    for i, v in enumerate(df_results["MAE ($)"]):
        axes[1].text(i, v + 5000, f"${v:,.0f}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "model_benchmark_comparison.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved model_benchmark_comparison.png")

    # 2. Predicted vs Actual Scatter Plots (4 Panels)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    models_preds = [
        ("Ridge Regression", y_pred_ridge, axes[0, 0], "#95a5a6"),
        ("Random Forest", y_pred_rf, axes[0, 1], "#7f8c8d"),
        ("Extra Trees", y_pred_et, axes[1, 0], "#3498db"),
        ("EBM (Glass-Box)", y_pred_ebm, axes[1, 1], "#2ecc71")
    ]
    
    min_val = 0.3 * 1e6
    max_val = 4.2 * 1e6
    
    for title, preds, ax, color in models_preds:
        ax.scatter(y_test / 1e6, preds / 1e6, alpha=0.55, color=color, edgecolors="none", s=32)
        ax.plot([min_val / 1e6, max_val / 1e6], [min_val / 1e6, max_val / 1e6], "r--", lw=2, label="Perfect Fit (y = x)")
        ax.set_title(f"{title} (R² = {r2_score(y_test, preds):.3f})", fontweight="bold")
        ax.set_xlabel("Actual Price ($ Millions AUD)")
        ax.set_ylabel("Predicted Price ($ Millions AUD)")
        ax.set_xlim(min_val / 1e6, max_val / 1e6)
        ax.set_ylim(min_val / 1e6, max_val / 1e6)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "predicted_vs_actual_all_models.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved predicted_vs_actual_all_models.png")

    # 3. Residual Distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    for title, preds, _, color in models_preds:
        residuals = (y_test - preds) / 1e3
        sns.kdeplot(residuals, label=f"{title} (Std: ${residuals.std():,.0f}k)", ax=ax, lw=2, color=color)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_title("Residual Error Distributions (y_true - y_pred) ($ Thousands AUD)", fontweight="bold")
    ax.set_xlabel("Residual Error ($ Thousands AUD)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "residual_error_distributions.png"), dpi=300)
    plt.close(fig)
    print("  -> Saved residual_error_distributions.png")

    return df_results


if __name__ == "__main__":
    train_and_benchmark_models()
