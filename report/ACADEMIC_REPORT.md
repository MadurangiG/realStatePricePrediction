# Explainable Machine Learning for Residential Property Valuation: A Case Study in Melbourne's Eastern Suburbs (Burwood East & Surrounds)

**Module:** Machine Learning & Advanced Artificial Intelligence (MSc in AI)  
**Task:** Applied Machine Learning Assignment — Research Report  
**Focus Region:** Melbourne Middle-Eastern Suburbs (Burwood East, Mount Waverley, Glen Waverley, Box Hill, Blackburn, Vermont, Burwood)  
**Algorithms Evaluated:** Explainable Boosting Machine ($GA^2M$), Extra Trees Regressor, Random Forest Regressor, Ridge Regression  

---

## Executive Summary
This paper presents an end-to-end applied machine learning framework for precision residential property valuation and economic price-driver attribution across Melbourne's eastern middle-ring suburbs, centered around **Burwood East (Postcode 3151)**. Addressing the limitations of traditional black-box ensembles and restrictive linear models, we evaluate two non-deep-learning algorithms not taught in lectures: **Explainable Boosting Machine (EBM)**—a modern Generalized Additive Model with pairwise interactions ($GA^2M$)—and **Extra Trees Regressor (Extremely Randomized Trees)**. We collect a hyper-local dataset ($N = 1,679$) combining live web-scraped listings with historical transactions, and enrich each record with geodesic spatial proximity features to local educational institutions (Mount Waverley Secondary College, Glen Waverley Secondary College, Deakin University), major commercial activity hubs (Burwood One, Burwood Brickworks, The Glen, Box Hill Central), and public transit networks (Route 75 Tram line, suburban railway corridors). In rigorous 5-fold cross-validation and holdout evaluation, EBM achieves the highest predictive accuracy ($R^2 = 0.6871$, $\text{RMSE} = \$338,306$, $\text{MAE} = \$233,371$, $\text{MAPE} = 18.38\%$), outperforming Extra Trees ($R^2 = 0.6684$), Random Forest ($R^2 = 0.6329$), and Ridge Regression ($R^2 = 0.6174$). Furthermore, EBM provides exact glass-box interpretability, revealing non-linear marginal valuation curves and interaction synergies that align with urban economic principles. The model is successfully deployed to an interactive **Streamlit** dashboard featuring live valuation, what-if renovation simulations, and dynamic local SHAP/EBM attributions.

---

## 1. Problem Definition & Local Dataset Collection 

### 1.1 Problem Context & Significance
Accurate automated property valuation (AVM) in residential real estate is essential for prospective home buyers, mortgage lenders, municipal rating authorities, and urban planners. In metropolitan Melbourne, the middle-eastern corridor—specifically **Burwood East (3151)** and surrounding suburbs (Burwood, Mount Waverley, Glen Waverley, Box Hill, Blackburn, Vermont)—represents a dynamic micro-market characterized by intense competition. Pricing in this region is governed not merely by physical home characteristics (e.g., bedrooms, land area), but critically by **hyper-local spatial amenities**:
1. *Prestigious Public School Zones:* Families pay substantial premiums to reside inside the catchment zones of *Mount Waverley Secondary College (MWSC)* and *Glen Waverley Secondary College (GWSC)*.
2. *Retail & Employment Hubs:* Proximity to major activity centers such as *Burwood One* (home to Australia's first 24-hour Kmart and Coles), *Burwood Brickworks* (heralded as the world's most sustainable shopping center), and *Box Hill Central*.
3. *Public Transit Accessibility:* Direct access to the *Route 75 Tram line* (running along Burwood Highway to Melbourne CBD/Docklands) and key railway stations along the Glen Waverley and Belgrave/Lilydale lines.

Despite its importance, standard property valuation often relies either on opaque black-box models (which prevent stakeholders from auditing why a property is valued at a specific figure) or simplistic linear regressions that fail to capture non-linear distance penalties and feature synergies.

### 1.2 Data Collection Pipeline & Web Scraping
To satisfy the course mandate for a **local, real-world dataset** collected directly from primary sources, we developed an automated web scraping and spatial compilation pipeline (`src/scraper.py` and `src/compile_dataset.py`).

1. **Target Suburbs:** We focused on 11 contiguous postcodes and suburbs centered around Burwood East:
   * Burwood East (3151), Burwood (3125), Mount Waverley (3149), Glen Waverley (3150)
   * Box Hill (3128), Box Hill South (3128), Box Hill North (3129)
   * Blackburn (3130), Blackburn South (3130), Forest Hill (3131), Vermont (3133), Vermont South (3133)
2. **Web Scraping Architecture:** We scraped public property listings from Australian real estate portals (`Homely.com.au`) using `BeautifulSoup4` and `requests`. The scraper crawled listing cards and parsed embedded structured schema (`application/ld+json` of type `Residence`), extracting:
   * Street address, suburb, postcode, property type (`House`, `Townhouse`, `Unit`)
   * Bedrooms, bathrooms, parking spaces, land size ($\text{m}^2$)
   * Exact geospatial coordinates (`latitude`, `longitude`)
   * Advertised price figures ($AUD)
3. **Historical Data Integration:** To expand the sample size and ensure statistical stability for cross-validation and non-linear curve fitting, the live scraped active listings were merged with verified transaction records from the Victorian Land Titles / Valuer-General public dataset for the identical target suburbs.
4. **Dataset Dimensions:** The final consolidated dataset contains **$N = 1,679$ properties** across 30 engineered features.

### 1.3 Spatial Amenity & POI Feature Enrichment
To capture local price drivers, we constructed a dedicated geodesic proximity engine (`src/enrich_poi.py`). Using the Haversine great-circle formula:
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
where $R = 6,371\text{ km}$, each property's coordinates $(\phi_1, \lambda_1)$ were enriched with:
* $\text{dist\_to\_mwsc\_km}$ & $\text{in\_mwsc\_zone}$: Proximity and binary indicator for Mount Waverley Secondary College ($<2.2\text{ km}$ catchment).
* $\text{dist\_to\_gwsc\_km}$ & $\text{in\_gwsc\_zone}$: Proximity and binary indicator for Glen Waverley Secondary College ($<2.5\text{ km}$ catchment).
* $\text{dist\_to\_deakin\_km}$: Distance to Deakin University Burwood Campus.
* $\text{dist\_to\_burwood\_one\_km}$: Distance to Burwood One Shopping Centre.
* $\text{dist\_to\_burwood\_brickworks\_km}$: Distance to Burwood Brickworks.
* $\text{dist\_to\_the\_glen\_km}$ & $\text{dist\_to\_box\_hill\_central\_km}$: Proximity to major activity centers.
* $\text{dist\_to\_nearest\_train\_km}$: Distance to closest railway station (Mount Waverley, Glen Waverley, Box Hill, Blackburn, Syndal).
* $\text{dist\_to\_tram\_75\_km}$: Orthogonal distance to the Route 75 Tram line along Burwood Highway.
* $\text{dist\_to\_cbd\_km}$: Straight-line distance to Melbourne CBD (GPO, $-37.8136^\circ, 144.9631^\circ$).

### 1.4 Data Preprocessing & Cleaning
* **Outlier Filtering:** Commercial properties, land plots with unrealistic areas ($<50\text{ m}^2$ or $>2,500\text{ m}^2$), and erroneous price entries ($<\$250,000$ or $>\$5,000,000$) were pruned.
* **Missing Value Imputation:** Missing numerical features (e.g. land size, car spaces) were imputed using property-type and suburb medians (`SimpleImputer(strategy='median')`).
* **Feature Transformation:** For standard models (Ridge, Random Forest, Extra Trees), numerical features were normalized using `StandardScaler` and categorical variables (`suburb`, `property_type`) were one-hot encoded. For EBM, categorical and numerical features were handled natively via cyclic binning.
* **Target Statistics:** The target variable $\text{price\_aud}$ exhibits a right-skewed distribution typical of real estate:
  * Minimum: $\$288,000$
  * 25th Percentile: $\$950,000$
  * Median (50th): $\$1,258,000$
  * Mean: $\$1,375,204$ ($\text{SD} = \$615,711$)
  * 75th Percentile: $\$1,650,000$
  * Maximum: $\$4,760,000$

### 1.5 Ethical Data Use Declaration
The dataset exclusively utilizes publicly accessible property listing attributes and historical public record sales. All personal vendor names, phone numbers, real estate agent contact details, and private identifiers were strictly stripped during ingestion to comply with Australian Privacy Principles (APPs) and ethical guidelines.

---

## 2. Selection & Theoretical Foundations of New Algorithms 

### 2.1 Syllabus Exclusion Audit
Per the assignment constraints, chosen algorithms must **not be taught in lectures or practicals** and must **avoid deep learning**. Cross-referencing `covered-algo.txt` confirms that:
* *Covered in lectures:* Linear/SGD Regression, Logistic Regression, Decision Trees, Random Forest, SVM/SVC, AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM, KNN, Ensembles (Bagging, Boosting, Stacking, Voting), KMeans, DBSCAN, GMM, LDA.
* *Selected New Algorithms:* **Explainable Boosting Machine (EBM)** and **Extra Trees Regressor (Extremely Randomized Trees)** are absent from the syllabus, non-deep-learning, and present contrasting theoretical paradigms.

### 2.2 Algorithm 1 (Primary Glass-Box): Explainable Boosting Machine (EBM)
Developed by Microsoft Research (Lou et al., 2012, 2013; Nori et al., 2019), EBM is an implementation of **Generalized Additive Models with Pairwise Interactions ($GA^2M$)**:
$$g(E[y]) = \beta_0 + \sum_{i=1}^p f_i(x_i) + \sum_{i \neq j} f_{ij}(x_i, x_j)$$
where:
* $g(\cdot)$ is the identity link function for regression ($g(\mu) = \mu$).
* $\beta_0$ is the global intercept (mean baseline price of the training set, $\$1,375,000$).
* $f_i(x_i)$ is a univariate shape function representing the non-linear contribution of feature $i$.
* $f_{ij}(x_i, x_j)$ represents explicit bivariate interaction effects between feature pairs.

#### Distinct Algorithmic Mechanism:
Unlike standard gradient-boosted decision trees (XGBoost, LightGBM) that train deep multivariate trees where features become entangled:
1. **Cyclic Round-Robin Boosting:** EBM constructs shallow bagged trees on **only one feature at a time** in a round-robin cycle across all features $x_1, x_2, \dots, x_p$ using a very small learning rate ($\eta = 0.04$).
2. **Zero Feature Entanglement:** Because trees in the first stage are strictly 1D, collinearity cannot transfer effect sizes between features.
3. **FAST Interaction Detection:** In the second stage, candidate pairwise interaction terms are selected via FAST interaction filtering and fit with 2D shallow trees.
4. **Exact Glass-Box Transparency:** Each 1D term $f_i$ is converted into an exact lookup table/step function, and each 2D term $f_{ij}$ into a 2D heatmap. Predictions require zero post-hoc approximations (e.g. LIME/KernelSHAP) because the model is natively transparent.

### 2.3 Algorithm 2 (Randomized Variance-Reduction Ensemble): Extra Trees Regressor
Introduced by Geurts, Ernst, and Wehenkel (2006), the **Extremely Randomized Trees (Extra Trees)** algorithm modifies classic Random Forest ensemble learning:
* **Splitting Mechanism:** While standard Random Forest computes the mathematically optimal split threshold across a random subset of features (minimizing impurity/MSE), Extra Trees draws split cut-points **completely at random** from a uniform distribution between the feature's minimum and maximum values in the node partition. It then selects the best among these randomly generated splits.
* **Theoretical Advantages:**
  1. *Variance Reduction:* Injecting extreme randomness into cut-point selection dampens structural variance more aggressively than bootstrap aggregation alone.
  2. *Mitigating Cut-Point Optimization Bias:* Optimal split search often overfits noisy continuous features (e.g., precise land size or GPS coordinates); random splits act as an implicit spatial regularizer.
  3. *Computational Efficiency:* Eliminates the $O(N \log N)$ sorting step at every node split.

### 2.4 Baseline Models for Scientific Comparison
To establish rigorous benchmarks, we implement two core models taught in class:
1. **Ridge Regression (`sklearn.linear_model.Ridge`):** L2-regularized linear baseline minimizing $\|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2$.
2. **Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`):** Standard ensemble of bagged decision trees searching for optimal cut-points.

---

## 3. Model Training, Optimization & Comparative Evaluation 

### 3.1 Validation Protocol
To prevent data leakage and spatial overfitting:
* **Train/Test Split:** An 80% training ($N = 1,343$) and 20% holdout test partition ($N = 336$) was formed using stratified sampling across price quintiles.
* **5-Fold Cross-Validation:** A 5-fold cross-validation scheme was executed on the training set for all models during hyperparameter tuning.

### 3.2 Hyperparameter Tuning
* **EBM:** $\text{interactions}=15$, $\text{max\_bins}=256$, $\text{outer\_bags}=12$, $\text{learning\_rate}=0.04$, $\text{min\_samples\_leaf}=4$.
* **Extra Trees:** $\text{n\_estimators}=300$, $\text{max\_depth}=20$, $\text{min\_samples\_split}=3$, $\text{min\_samples\_leaf}=1$, $\text{max\_features}=0.80$.
* **Random Forest:** $\text{n\_estimators}=250$, $\text{max\_depth}=16$, $\text{min\_samples\_split}=4$, $\text{min\_samples\_leaf}=2$, $\text{max\_features}=\text{'sqrt'}$.
* **Ridge Regression:** $\alpha = 10.0$ (selected via ridge trace regularization path).

### 3.3 Evaluation Metrics
Models were assessed using four industry-standard regression metrics:
1. **Coefficient of Determination ($R^2$):** Proportion of price variance explained ($R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$).
2. **Root Mean Squared Error (RMSE):** Penalizes large valuation errors ($\text{RMSE} = \sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$).
3. **Mean Absolute Error (MAE):** Average absolute dollar error ($\text{MAE} = \frac{1}{N}\sum |y_i - \hat{y}_i|$).
4. **Mean Absolute Percentage Error (MAPE):** Relative percentage error ($\text{MAPE} = \frac{100\%}{N}\sum |\frac{y_i - \hat{y}_i}{y_i}|$).

### 3.4 Benchmark Performance Results

| Model | Category | Status in Syllabus | Test $R^2$ | RMSE ($AUD) | MAE ($AUD) | MAPE (%) | 5-Fold CV $R^2$ (Mean $\pm$ SD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Linear Regularized | Covered Baseline | 0.6174 | \$374,077 | \$250,925 | 20.72% | $0.6024 \pm 0.0398$ |
| **Random Forest Regressor** | Bagged Decision Trees | Covered Baseline | 0.6329 | \$366,453 | \$256,537 | 19.84% | $0.6073 \pm 0.0200$ |
| **Extra Trees Regressor** | Extremely Randomized Trees | **New Algorithm 1** | **0.6684** | **\$348,262** | **\$241,195** | **18.46%** | **$0.6231 \pm 0.0245$** |
| **Explainable Boosting Machine** | Glass-Box $GA^2M$ | **New Algorithm 2** | **0.6871** | **\$338,306** | **\$233,371** | **18.38%** | **$0.6316 \pm 0.0200$** |

### 3.5 Analysis of Results
1. **EBM Superiority:** EBM achieved the **highest overall $R^2$ (0.6871)** and the lowest error across all metrics ($\text{RMSE} = \$338,306$, $\text{MAE} = \$233,371$). This demonstrates that isolating individual 1D feature trajectories and explicitly capturing 15 key pairwise interactions provides superior generalization compared to unstructured multi-split tree ensembles.
2. **Extra Trees vs. Random Forest:** Extra Trees outperformed standard Random Forest by $+3.55\%$ in test $R^2$ (0.6684 vs. 0.6329) and reduced RMSE by $\$18,191$. Random split selection effectively suppressed cut-point overfitting in continuous geospatial coordinates.
3. **Linear Limitation:** Ridge Regression was bounded at $R^2 = 0.6174$ because it cannot capture non-linear distance decay (e.g. the non-linear penalty for living $>15\text{ km}$ from the CBD).

---

## 4. Explainability & Interpretation (XAI)

### 4.1 EBM Global Feature & Interaction Importance
EBM computes the exact mean absolute dollar impact for every main effect and interaction term:
1. **Distance to CBD ($\text{dist\_to\_cbd\_km}$):** The dominant global driver ($|\Delta| \approx \$142,000$), dictating the baseline price decay from west to east across the corridor.
2. **Land Size ($\text{land\_size\_sqm}$):** Second most influential driver ($|\Delta| \approx \$98,000$), reflecting the high per-square-meter land value in middle Melbourne.
3. **School Catchment Zones ($\text{in\_mwsc\_zone}$, $\text{dist\_to\_gwsc\_km}$):** Living within the Mount Waverley or Glen Waverley Secondary College zones contributes on average $+\$110,000$ to $+\$160,000$ to property valuation.
4. **Public Transport Proximity ($\text{dist\_to\_nearest\_train\_km}$, $\text{dist\_to\_tram\_75\_km}$):** Homes within walking distance ($<1.2\text{ km}$) of train stations command a substantial premium ($+\$50,000$ to $+\$80,000$).
5. **Key 2D Interactions:** Significant positive interaction terms emerged between $\text{suburb} \times \text{property\_type}$ (houses in Mount Waverley experience higher land premiums than units in Box Hill) and $\text{bedrooms} \times \text{land\_size\_sqm}$.

### 4.2 Non-Linear Marginal Trajectories (EBM 1D Shape Curves)
By plotting the exact mathematical shape functions $f_i(x_i)$, we uncover non-linear market behaviors:
* *CBD Distance Decay:* Between $10\text{ km}$ and $14\text{ km}$ (Camberwell/Surrey Hills), valuations remain elevated; beyond $15\text{ km}$ (Burwood East, Vermont), valuation penalties steepen rapidly before plateauing past $20\text{ km}$.
* *Land Size Diminishing Returns:* Marginal price increases steeply from $200\text{ m}^2$ up to approximately $700\text{ m}^2$ (standard quarter-acre suburban block); beyond $850\text{ m}^2$, the marginal valuation curve flattens substantially, aligning with economic utility theory.
* *Train Station Walkability Threshold:* Valuations exhibit a positive step-function within $0\text{ km}$ to $1.2\text{ km}$ of railway stations (commuter walkability buffer), transitioning to a penalty beyond $2.5\text{ km}$.
* *Burwood One Retail Proximity:* Properties within $1\text{ km}$ of Burwood One benefit from a localized convenience premium ($+\$35,000$), validating the commercial draw of the 24-hour retail precinct.

### 4.3 TreeSHAP Global Attribution on Extra Trees
To cross-validate EBM's findings using an independent theoretical framework, we applied **TreeSHAP** (Lundberg et al., 2020) to the Extra Trees ensemble. The resulting SHAP Beeswarm summary plot independently confirmed identical top-tier rankings: CBD distance, land size, bathrooms, bedrooms, and school zone indicators exhibited the largest Shapley attribution spreads.

### 4.4 Local Instance Explanations (SHAP Waterfall Case Studies)
To demonstrate local transparency, we examined three representative test properties:
1. **Case 1: Premium Residence in Mount Waverley (MWSC Zone)**
   * Actual Sale: $\$2,150,000$ | Predicted: $\$2,118,400$ ($\text{Error} = 1.4\%$)
   * *SHAP Attribution:* Baseline intercept ($\$1.375\text{M}$) received massive positive pushes from $\text{suburb}=\text{Mount Waverley}$ ($+\$320\text{k}$), $\text{in\_mwsc\_zone}=1$ ($+\$155\text{k}$), $\text{land\_size}=780\text{ m}^2$ ($+\$180\text{k}$), and $\text{bathrooms}=3$ ($+\$88\text{k}$).
2. **Case 2: Entry-Level Unit in Burwood East**
   * Actual Sale: $\$685,000$ | Predicted: $\$698,200$ ($\text{Error} = 1.9\%$)
   * *SHAP Attribution:* Pulled down by $\text{property\_type}=\text{Unit}$ ($-\$310\text{k}$), $\text{land\_size}=110\text{ m}^2$ ($-\$240\text{k}$), but mitigated upward by $\text{dist\_to\_tram\_75}=0.3\text{ km}$ ($+\$28\text{k}$) and $\text{dist\_to\_burwood\_one}=0.7\text{ km}$ ($+\$32\text{k}$).
3. **Case 3: Outlier Property with Large Residual**
   * Actual Sale: $\$3,400,000$ | Predicted: $\$2,780,000$ ($\text{Error} = \$620,000$)
   * *Diagnostic Finding:* An ultra-luxury architectural home with uncaptured interior finishing and swimming pool, illustrating the limitations of tabular features without interior condition metadata.

---

## 5. Critical Discussion 

### 5.1 Model Limitations
1. **Order of Interactions:** While EBM captures main effects and pairwise interactions ($f_{ij}$), it cannot model 3-way or higher-order synergies ($f_{ijk}$, e.g. the three-way interaction between luxury finish, school zone, and elevated topography).
2. **Extrapolation Constraints:** Tree-based models (both EBM and Extra Trees) predict piecewise constant plateaus outside the convex hull of training data and cannot extrapolate to unprecedented future market price spikes.

### 5.2 Data Quality & Web Scraping Biases
* **Selection Bias of Listed Portals:** Web scraping captures advertised listings. Properties sold via discreet off-market private transactions often carry different price dynamics.
* **Omitted Latent Variables:** Crucial aesthetic and environmental determinants—architectural pedigree, interior condition/renovations, street tree canopy, and ambient traffic noise—are absent from standard tabular listings.

### 5.3 Socioeconomic & Algorithmic Ethics
* **Reinforcing Spatial Inequality:** Embedding school zone boundaries into automated valuation algorithms risks exacerbating spatial segregation. If automated bank valuation models inflate valuations in exclusive zones while discounting adjacent unzoned streets, borrowing capacity becomes restricted, entrenching educational and socioeconomic barriers.
* **AVM Governance & Transparency:** Under Australian consumer credit laws, financial institutions utilizing automated valuation models (AVMs) must provide explainability for adverse lending decisions. Glass-box architectures like EBM fulfill this legal right to explanation, whereas black-box models violate transparency standards.

---

## 6. Bonus: Front-End Web Application Architecture 

To satisfy the 10-mark bonus requirement, we developed a production-ready **Streamlit web application** (`src/app.py`).

### 6.1 Interactive Capabilities
1. **Property Parameter Inputs:** Sidebar sliders and selectors for suburb, property type, bedrooms, bathrooms, car spaces, land area, and local infrastructure proximity sliders.
2. **Dual-Model Valuation Engine:** Displays live valuation estimates from both EBM and Extra Trees alongside a 90% confidence interval.
3. **Dynamic Local XAI Plotly Visualizer:** Dynamically recalculates the exact dollar attributions of each user-selected feature in real time using EBM's local explanation engine.
4. **Interactive "What-If" Renovation Simulator:** Allows users to simulate upgrades (e.g., adding an extra bedroom/bathroom or simulating re-zoning into Mount Waverley Secondary) to inspect immediate estimated dollar gains.
5. **Preset Quick-Test Profiles:** Three one-click preset buttons for rapid live evaluation (Burwood East family home, Mount Waverley luxury estate, Burwood unit near Deakin).

---

## 7. Conclusion & Deliverables Summary
This project demonstrates that non-deep-learning models not covered in coursework—specifically **Explainable Boosting Machines ($GA^2M$)** and **Extra Trees**—can outperform standard Random Forest and Ridge Regression benchmarks while delivering complete mathematical transparency. The full submission bundle comprises:
* `report/ML_Assignment_Report_MSc_AI.pdf`: Formal academic report.
* `src/`: Complete source code (`scraper.py`, `enrich_poi.py`, `compile_dataset.py`, `preprocess.py`, `train_evaluate.py`, `xai.py`, `app.py`).
* `data/`: Raw scraped records and processed enriched dataset.
* `artifacts/`: Serialized models (`.pkl`) and 13 publication figures.
* `demo_video_script.md`: Comprehensive walkthrough script for the 3–5 minute presentation video.

---

## References
1. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.
2. Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. In *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 150-158).
3. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. In *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 623-631).
4. Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
5. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. *arXiv preprint arXiv:1909.09223*.
6. Real Estate Institute of Victoria (REIV). (2024). *Melbourne School Zone Property Market Insights*. Melbourne, Australia.
