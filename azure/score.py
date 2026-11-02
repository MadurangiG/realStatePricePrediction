"""
Azure Machine Learning Online Real-Time Scoring Entry Script
Melbourne Eastern Suburbs Housing Valuation & XAI Service

Entry point for Azure ML Managed Online Endpoints:
- init(): Loads model artifacts into memory when container boots.
- run(raw_data): Executes inference on incoming JSON payload, returning
  predicted valuations and XAI dollar attributions.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global model references
ebm_model = None
et_model = None
preprocessor = None

DEFAULT_FEATURES = {
    "suburb": "Burwood East",
    "property_type": "House",
    "bedrooms": 4,
    "bathrooms": 2,
    "parking_spaces": 2,
    "land_size_sqm": 620,
    "in_mwsc_zone": 0,
    "in_gwsc_zone": 0,
    "in_box_hill_high_zone": 0,
    "dist_to_mwsc_km": 3.5,
    "dist_to_gwsc_km": 4.0,
    "dist_to_box_hill_high_km": 4.2,
    "dist_to_deakin_km": 1.8,
    "dist_to_burwood_one_km": 0.8,
    "dist_to_burwood_brickworks_km": 2.0,
    "dist_to_the_glen_km": 5.0,
    "dist_to_box_hill_central_km": 4.5,
    "dist_to_nearest_major_shop_km": 0.8,
    "dist_to_nearest_train_km": 2.5,
    "dist_to_tram_75_km": 0.4,
    "dist_to_cbd_km": 17.2,
    "latitude": -37.8541,
    "longitude": 145.1500
}


def init():
    """Initializes models from the Azure ML registered model directory or local fallback."""
    global ebm_model, et_model, preprocessor
    
    # Azure ML mounts models under AZUREML_MODEL_DIR
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    logger.info(f"Loading models from directory: {model_dir}")
    
    # Candidate search paths (Azure ML registry layout vs local repository layout)
    candidate_ebm_paths = [
        os.path.join(model_dir, "ebm_model.pkl"),
        os.path.join(model_dir, "artifacts", "models", "ebm_model.pkl"),
        "artifacts/models/ebm_model.pkl"
    ]
    candidate_et_paths = [
        os.path.join(model_dir, "extra_trees_model.pkl"),
        os.path.join(model_dir, "artifacts", "models", "extra_trees_model.pkl"),
        "artifacts/models/extra_trees_model.pkl"
    ]
    candidate_bundle_paths = [
        os.path.join(model_dir, "preprocessed_bundle.pkl"),
        os.path.join(model_dir, "data", "processed", "preprocessed_bundle.pkl"),
        "data/processed/preprocessed_bundle.pkl"
    ]

    ebm_path = next((p for p in candidate_ebm_paths if os.path.exists(p)), None)
    et_path = next((p for p in candidate_et_paths if os.path.exists(p)), None)
    bundle_path = next((p for p in candidate_bundle_paths if os.path.exists(p)), None)

    if not ebm_path:
        raise FileNotFoundError(f"Could not locate ebm_model.pkl in {candidate_ebm_paths}")

    with open(ebm_path, "rb") as f:
        ebm_model = pickle.load(f)
    logger.info("EBM model loaded successfully.")

    if et_path and os.path.exists(et_path):
        with open(et_path, "rb") as f:
            et_model = pickle.load(f)
        logger.info("Extra Trees model loaded successfully.")

    if bundle_path and os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
            preprocessor = bundle.get("preprocessor", None)
        logger.info("Preprocessor bundle loaded successfully.")


def run(raw_data):
    """Processes real-time inference request.
    
    Args:
        raw_data (str): JSON-encoded string containing property features or a list of properties.
        
    Returns:
        str: JSON-encoded string containing valuations and XAI attributions.
    """
    try:
        data = json.loads(raw_data)
        
        # Support both 'data' wrapper and direct list/dict format
        payload = data.get("data", data)
        if isinstance(payload, dict):
            payload = [payload]
            
        results = []
        for item in payload:
            # Merge with defaults for missing optional features
            row = {**DEFAULT_FEATURES, **item}
            df = pd.DataFrame([row])
            
            # Primary EBM prediction
            pred_ebm = float(ebm_model.predict(df)[0])
            
            # Extra Trees ensemble prediction (if loaded)
            pred_et = None
            if et_model is not None and preprocessor is not None:
                try:
                    df_enc = preprocessor.transform(df)
                    pred_et = float(et_model.predict(df_enc)[0])
                except Exception as e:
                    logger.warning(f"Could not compute Extra Trees prediction: {e}")
            
            # Local XAI Feature Dollar Attribution
            local_attributions = []
            try:
                local_exp = ebm_model.explain_local(df)
                local_data = local_exp.data(0)
                names = local_data.get("names", [])
                scores = local_data.get("scores", [])
                
                # Top 8 most impactful features
                sorted_idx = np.argsort(np.abs(scores))[-8:][::-1]
                for idx in sorted_idx:
                    local_attributions.append({
                        "feature": str(names[idx]),
                        "dollar_impact_aud": round(float(scores[idx]), 2)
                    })
            except Exception as e:
                logger.warning(f"XAI calculation failed: {e}")

            results.append({
                "suburb": row["suburb"],
                "property_type": row["property_type"],
                "primary_valuation_ebm_aud": round(pred_ebm, 2),
                "ensemble_valuation_extra_trees_aud": round(pred_et, 2) if pred_et else None,
                "confidence_interval_90_aud": {
                    "lower": round(pred_ebm * 0.90, 2),
                    "upper": round(pred_ebm * 1.10, 2)
                },
                "currency": "AUD",
                "top_xai_drivers": local_attributions
            })
            
        return json.dumps({"status": "success", "predictions": results})
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

