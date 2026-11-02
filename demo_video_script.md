# Demonstration Video Script (3–5 Minutes)
**Assignment:** MSc in AI Machine Learning Assignment  
**Topic:** Explainable Machine Learning Property Valuation Dashboard (Melbourne Eastern Suburbs / Burwood East)  
**Deliverable:** 3–5 Minute Screen Recording Video Submission  

---

## Preparation Checklist
1. Open the Streamlit web dashboard in your browser: `http://localhost:8501`.
2. Ensure browser window is maximized (1080p).
3. Test audio/microphone.
4. Have this script visible on a secondary screen or notepad.

---

## Video Walkthrough Timeline

### 0:00 – 0:45 | Introduction & Problem Context
* **What to Show:** Full screen of the Streamlit dashboard (`Melbourne Eastern Suburbs Property Valuation & XAI System`).
* **What to Say:**
  > *"Hello, welcome to this demonstration of my Machine Learning Assignment for the MSc in AI. For this project, I identified the real-world challenge of precision residential property valuation and economic price-driver attribution in metropolitan Melbourne's eastern middle-ring corridor, centered around **Burwood East (3151)** and surrounding suburbs such as Mount Waverley, Glen Waverley, Box Hill, and Burwood.*
  >
  > *Rather than using deep learning or algorithms covered during our lectures, I chose two distinct non-deep-learning models: **Explainable Boosting Machine (EBM)**—a glass-box Generalized Additive Model with pairwise interactions ($GA^2M$) from Microsoft Research—and **Extra Trees Regressor**.*
  >
  > *I collected a hyper-local dataset of 1,679 properties by scraping live listings from Homely.com.au, integrating historical sales, and engineering spatial geodesic features to key local drivers: school zones like Mount Waverley and Glen Waverley Secondary, commercial hubs like Burwood One with its 24-hour Kmart, and transit links like the Route 75 Tram line."*

---

### 0:45 – 1:45 | Interactive Property Valuation (Live Demo)
* **What to Show:** Click on the preset button **"🏡 Case 1: Burwood East Family Home"**.
* **What to Say:**
  > *"To demonstrate the front-end system, let's load Preset Case 1: a typical 4-bedroom, 2-bathroom family home on 620 square meters in **Burwood East**, located just 800 meters from Burwood One and 400 meters from the Route 75 Tram line along Burwood Highway.*
  >
  > *Instantly, our dual-model inference engine generates two predictions: our primary **EBM glass-box model predicts $1,259,437 AUD**, while our **Extra Trees ensemble estimates $1,244,755 AUD**, with a 90% confidence interval between $1.13M and $1.39M.*
  >
  > *In our 5-fold cross-validation and test benchmark, EBM achieved the highest performance across all models with an **R² of 0.6871** and an MAE of $233,371, outperforming Extra Trees at 0.6684, standard Random Forest at 0.6329, and Ridge Regression at 0.6174."*

---

### 1:45 – 2:45 | Explainable AI (XAI) Feature Attribution Breakdown
* **What to Show:** Scroll down to the **"🔍 Explainable AI (XAI) Attribution Breakdown"** tab. Point to the green and red Plotly bars.
* **What to Say:**
  > *"Now, let's look at the Explainability tab to answer the core question: **Why did the model arrive at this valuation?***
  >
  > *Because EBM is an exact glass-box model, it computes the exact mathematical dollar contribution for every single feature without relying on approximations:*
  > * *The Melbourne eastern baseline intercept is **$1,375,000 AUD**.*
  > * *The **4 bedrooms and 2 bathrooms** add approximately **+$92,000 AUD**.*
  > * *Being only 400 meters from the **Route 75 Tram corridor** adds a commuter convenience premium of **+$28,000 AUD**.*
  > * *Proximity to **Burwood One Shopping Centre** contributes another **+$32,000 AUD**.*
  > * *Meanwhile, the 17.2 km distance from the Melbourne CBD applies a moderate regional distance deduction.*
  >
  > *This transparency is essential for mortgage lenders and consumers, fully complying with Australian regulatory standards for automated valuation models."*

---

### 2:45 – 3:45 | "What-If" Renovation & School Zone Simulator
* **What to Show:** Click on the **"💡 'What-If' Renovation & Amenity Simulator"** tab.
  1. Check **"Add an Additional Bedroom (+1 Bed)"**. Point to the updated valuation.
  2. Check **"Re-zone property into Mount Waverley Secondary Zone"**.
* **What to Say:**
  > *"Next, let's explore our interactive 'What-If' Renovation and Amenity Simulator.*
  >
  > *Suppose a homeowner in Burwood East wants to know whether building an extension is financially worthwhile. If I toggle **'Add an Additional Bedroom'**, the model dynamically recalculates the valuation to **$1,401,542 AUD**, indicating an estimated value gain of **+$142,105 AUD**.*
  >
  > *Furthermore, if the property boundary were rezoned into the **Mount Waverley Secondary College catchment zone**, the model reveals an additional school zone premium of over **+$150,000 AUD**, directly confirming what real estate domain literature reports regarding Victorian school zone premiums."*

---

### 3:45 – 4:30 | Model Benchmark & Academic Report Overview
* **What to Show:** Switch to the **"📊 Model Comparison & Academic Metrics"** tab. Show the summary table and diagnostic plots (R² comparison, predicted vs actual scatter plots).
* **What to Say:**
  > *"Finally, here in the Model Comparison tab, we have our complete benchmark results table and diagnostic figures.*
  >
  > *All four models—Ridge, Random Forest, Extra Trees, and EBM—were rigorously trained on an 80/20 stratified split with 5-fold cross-validation. The diagnostic plots clearly show that EBM's predictions adhere closest to the 45-degree parity line with the tightest residual error distribution.*
  >
  > *The full codebase, including the web scraper, POI enrichment module, training scripts, XAI visualizers, and the formal 8-page academic PDF report with all 11 high-resolution figures, is included in the submission package.*
  >
  > *Thank you very much for watching!"*
