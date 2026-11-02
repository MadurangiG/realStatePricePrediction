# Melbourne Eastern Suburbs Housing Valuation & Explainable AI (XAI) System

An end-to-end applied machine learning framework for precision residential property valuation and economic price-driver attribution across Melbourne's eastern middle-ring suburbs, centered around **Burwood East (Postcode 3151)**.

---

## 🌟 Key Project Highlights
- **Hyper-Local Dataset ($N = 1,679$):** Web-scraped listings merged with verified historical sales for Burwood East, Mount Waverley, Glen Waverley, Box Hill, Burwood, Blackburn, and Vermont.
- **Geodesic Spatial POI Feature Enrichment:** Great-circle distances to top schools (*MWSC*, *GWSC*, *Deakin Uni*), commercial hubs (*Burwood One*, *Burwood Brickworks*, *The Glen*, *Box Hill Central*), and transit networks (*Route 75 Tram line*, railway stations).
- **Two "New" Non-Deep-Learning Algorithms:**
  1. **Explainable Boosting Machine (EBM / $GA^2M$)** from Microsoft Research (*Primary glass-box model*).
  2. **Extra Trees Regressor (Extremely Randomized Trees)** (*Ensemble benchmark*).
- **Baseline Comparison (Covered in Course):** Standard **Random Forest Regressor** and **Ridge Regression**.
- **Model Benchmark Results:**
  * **EBM ($GA^2M$):** $R^2 = \mathbf{0.6871}$ | $\text{RMSE} = \mathbf{\$338,306}$ | $\text{MAE} = \mathbf{\$233,371}$ | $\text{MAPE} = \mathbf{18.38\%}$
  * **Extra Trees:** $R^2 = 0.6684$ | $\text{RMSE} = \$348,262$ | $\text{MAE} = \$241,195$
  * **Random Forest:** $R^2 = 0.6329$ | $\text{RMSE} = \$366,453$ | $\text{MAE} = \$256,537$
  * **Ridge Regression:** $R^2 = 0.6174$ | $\text{RMSE} = \$374,077$ | $\text{MAE} = \$250,925$
- **Explainability (XAI):** Global feature importances, 1D non-linear marginal shape curves, TreeSHAP beeswarm summary plot, and local instance waterfall explanations.
- **Bonus Front-End Integration (+10 Marks):** Production-ready **Streamlit web dashboard** (`src/app.py`) with real-time valuation, dynamic Plotly XAI attribution waterfall, and an interactive "What-If" renovation simulator.
- **Formal Academic Report:** Publication-grade PDF report (`report/ML_Assignment_Report_MSc_AI.pdf`) with 11 high-res figures and complete academic analysis.

---

## 📁 Repository Structure
```
realStatePricePrediction/
├── data/
│   ├── raw/
│   │   └── melbourne_housing_raw.csv        # Raw scraped listings from Homely
│   └── processed/
│       ├── melbourne_burwood_east_housing.csv # Master enriched dataset (N = 1,679)
│       ├── train_raw.csv / test_raw.csv     # 80/20 train and test splits
│       └── preprocessed_bundle.pkl          # Serialized arrays & preprocessor
├── src/
│   ├── scraper.py                           # Automated web scraper for Homely listings
│   ├── enrich_poi.py                        # Geodesic Haversine distance engine
│   ├── compile_dataset.py                   # Merges scraped & historical sales with POIs
│   ├── preprocess.py                        # Missing imputation, scaling, EDA generator
│   ├── train_evaluate.py                    # Trains Ridge, RF, Extra Trees, EBM (5-fold CV)
│   ├── xai.py                               # Computes EBM shapes & TreeSHAP plots
│   └── app.py                               # Interactive Streamlit dashboard (+10 Bonus)
├── artifacts/
│   ├── figures/                             # 13 high-resolution publication figures
│   ├── models/                              # Serialized model weights (.pkl)
│   └── model_benchmark_results.csv          # Evaluation metrics summary table
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
py -m pip install -r requirements.txt
```

### 2. Run Pipeline End-to-End
```bash
# Step 1: Scrape & compile master dataset
py src/compile_dataset.py

# Step 2: Run EDA & data preprocessing
py src/preprocess.py

# Step 3: Train, tune & benchmark all 4 models
py src/train_evaluate.py

# Step 4: Generate XAI interpretability figures
py src/xai.py

# Step 5: Compile formal academic PDF report
py report/generate_pdf.py
```

### 3. Launch Front-End Web Application (Bonus)
```bash
py -m streamlit run src/app.py
```
Open **`http://localhost:8501`** in your browser to interact with the dashboard!
