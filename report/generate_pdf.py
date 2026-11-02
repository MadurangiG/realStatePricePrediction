"""
PDF Generator for MSc in AI Machine Learning Assignment Report
Uses ReportLab to compile a publication-grade academic PDF report
embedding all performance benchmark tables and high-resolution figures.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)


def build_academic_pdf(output_pdf_path: str = "report/ML_Assignment_Report_MSc_AI.pdf"):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3799"),
        spaceAfter=8,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4b6584"),
        spaceAfter=14,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0c2461"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e3799"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2f3542"),
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#57606f"),
        alignment=1,
        spaceAfter=10
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e272e"),
        backColor=colors.HexColor("#f1f2f6"),
        borderColor=colors.HexColor("#70a1ff"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Explainable Machine Learning for Residential Property Valuation", title_style))
    story.append(Paragraph("<b>A Case Study in Melbourne's Eastern Suburbs (Burwood East & Surrounds)</b>", subtitle_style))
    story.append(Paragraph("<b>Course:</b> MSc in Artificial Intelligence & Machine Learning &nbsp;|&nbsp; <b>Module:</b> Advanced Machine Learning &nbsp;|&nbsp; <b>Evaluation:</b> Assignment Report", caption_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3799"), spaceBefore=4, spaceAfter=10))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary:</b> This study presents an applied machine learning framework for precision residential property valuation "
        "and economic price-driver attribution across Melbourne's eastern middle-ring suburbs, centered around <b>Burwood East (3151)</b>. "
        "We evaluate two non-deep-learning algorithms not taught in lectures: <b>Explainable Boosting Machine (EBM / GA²M)</b>—a modern Generalized Additive Model with pairwise interactions—"
        "and <b>Extra Trees Regressor (Extremely Randomized Trees)</b>. A hyper-local dataset (N = 1,679) was compiled combining live web-scraped listings with historical sales, "
        "and enriched with geodesic spatial proximity features to schools (Mount Waverley Secondary, Glen Waverley Secondary, Deakin University), commercial activity hubs (Burwood One, Burwood Brickworks), "
        "and transit corridors (Route 75 Tram line, railway stations). In rigorous 5-fold cross-validation and holdout evaluation, <b>EBM achieved the highest predictive performance "
        "(R² = 0.6871, RMSE = $338,306, MAE = $233,371, MAPE = 18.38%)</b>, outperforming Extra Trees (R² = 0.6684), Random Forest (R² = 0.6329), and Ridge Regression (R² = 0.6174). "
        "EBM provides exact glass-box interpretability, revealing non-linear marginal trajectories and interaction synergies that align with urban economic principles. "
        "The model is deployed to an interactive <b>Streamlit web application</b> featuring real-time valuation, what-if renovation simulations, and dynamic local XAI explanations."
    )
    story.append(Paragraph(summary_text, callout_style))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # Section 1: Problem Definition & Local Dataset Collection
    # ---------------------------------------------------------
    story.append(Paragraph("1. Problem Definition & Hyper-Local Dataset Collection (15 Marks)", h1_style))
    sec1_text = (
        "<b>1.1 Context & Relevance:</b> Accurate Automated Valuation Models (AVMs) in residential real estate are critical for prospective homebuyers, "
        "lending institutions, and municipal rating bodies. In Melbourne's eastern suburbs—specifically <b>Burwood East (Postcode 3151)</b>, Burwood, Mount Waverley, "
        "Glen Waverley, Box Hill, Blackburn, and Vermont—property pricing is driven by both physical home attributes and critical <b>spatial amenity drivers</b>: "
        "(i) school catchment zones (e.g. Mount Waverley Secondary College / MWSC and Glen Waverley Secondary College / GWSC), "
        "(ii) retail hubs (Burwood One featuring Australia's first 24h Kmart/Coles, Burwood Brickworks, The Glen, Box Hill Central), and "
        "(iii) transit corridors (Route 75 Tram line along Burwood Hwy to CBD and suburban railway lines).<br/><br/>"
        "<b>1.2 Web Scraping Architecture:</b> To compile an authentic, primary local dataset, an automated web scraper (<code>src/scraper.py</code>) "
        "was engineered using <code>BeautifulSoup4</code> and <code>requests</code>. The scraper targeted public listing cards on Australian real estate portal "
        "<code>Homely.com.au</code> across 11 target suburbs, parsing embedded structured <code>application/ld+json</code> records to extract exact coordinates "
        "(latitude, longitude), addresses, bedrooms, bathrooms, car spaces, and price figures. Live scraped listings were merged with historical public transaction records "
        "from the Victorian Land Titles registry, yielding a robust sample of <b>N = 1,679 unique properties</b>.<br/><br/>"
        "<b>1.3 POI Spatial Enrichment:</b> Using the Haversine great-circle formula (<code>src/enrich_poi.py</code>), every property was enriched with geodesic distances to: "
        "MWSC, GWSC, Box Hill High, Deakin University, Burwood One, Burwood Brickworks, The Glen, Box Hill Central, closest train station, Route 75 Tram corridor, and Melbourne CBD.<br/><br/>"
        "<b>1.4 Data Cleaning & Ethical Compliance:</b> Missing features were imputed via suburb/property-type medians. Outliers (land area &lt; 50 or &gt; 2500 m², price &lt; $250k or &gt; $5.0M) "
        "were filtered. All personal vendor, agent, and contact identifiers were stripped to ensure strict ethical data use under Australian Privacy Principles."
    )
    story.append(Paragraph(sec1_text, body_style))

    # Embed EDA Figures
    if os.path.exists("artifacts/figures/eda_price_distribution.png"):
        story.append(Image("artifacts/figures/eda_price_distribution.png", width=6.8 * inch, height=2.6 * inch))
        story.append(Paragraph("<b>Figure 1:</b> Distribution of residential property sale prices in Melbourne's Eastern Suburbs (Raw $ Millions AUD and Log10 transformed).", caption_style))

    if os.path.exists("artifacts/figures/eda_suburb_median_prices.png"):
        story.append(Image("artifacts/figures/eda_suburb_median_prices.png", width=6.2 * inch, height=3.3 * inch))
        story.append(Paragraph("<b>Figure 2:</b> Median property prices across Burwood East and surrounding Eastern Melbourne suburbs.", caption_style))

    if os.path.exists("artifacts/figures/eda_school_zone_premiums.png"):
        story.append(Image("artifacts/figures/eda_school_zone_premiums.png", width=6.8 * inch, height=2.8 * inch))
        story.append(Paragraph("<b>Figure 3:</b> Significant property price premiums observed for homes inside Mount Waverley Secondary (MWSC) and Glen Waverley Secondary (GWSC) catchment zones.", caption_style))

    if os.path.exists("artifacts/figures/eda_correlation_heatmap.png"):
        story.append(Image("artifacts/figures/eda_correlation_heatmap.png", width=6.2 * inch, height=4.5 * inch))
        story.append(Paragraph("<b>Figure 4:</b> Pearson correlation matrix demonstrating relationships between property price, structural attributes, and POI spatial amenities.", caption_style))

    story.append(PageBreak())

    # ---------------------------------------------------------
    # Section 2: Selection of New ML Algorithms
    # ---------------------------------------------------------
    story.append(Paragraph("2. Selection & Theoretical Justification of New ML Algorithms (15 Marks)", h1_style))
    sec2_text = (
        "<b>2.1 Syllabus Exclusion Verification:</b> The module covered Linear/SGD Regression, Logistic Regression, Decision Trees, Random Forest, "
        "SVM/SVC, AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM, KNN, Ensembles (Bagging, Boosting, Stacking, Voting), KMeans, and DBSCAN. "
        "Deep learning is strictly prohibited. To satisfy requirements, we chose two powerful non-deep-learning algorithms absent from the syllabus:<br/><br/>"
        "<b>2.2 Explainable Boosting Machine (EBM / GA²M) (Primary Algorithm):</b><br/>"
        "Developed by Microsoft Research (Lou et al., 2012, 2013; Nori et al., 2019), EBM is a modern implementation of Generalized Additive Models with Pairwise Interactions:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>y = β₀ + Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ, xⱼ)</b><br/>"
        "<i>Algorithmic Differentiation:</i> Traditional gradient-boosted trees (XGBoost/LightGBM) grow deep multivariate trees where features become entangled, creating opaque black boxes. "
        "In contrast, EBM trains shallow bagged decision trees on <b>only one feature at a time in a round-robin cycle</b> with a very small learning rate (η = 0.04). "
        "In a second pass, it detects and fits explicit 2D interaction terms fᵢⱼ(xᵢ, xⱼ). Each term fᵢ is an exact 1D lookup table/step function, producing <b>exact glass-box interpretability</b> "
        "with zero post-hoc approximation error while matching black-box ensemble accuracy.<br/><br/>"
        "<b>2.3 Extra Trees Regressor (Extremely Randomized Trees) (Ensemble Comparison):</b><br/>"
        "Introduced by Geurts et al. (2006), Extra Trees modifies Random Forest. While Random Forest computes the optimal split threshold across feature subsets, "
        "Extra Trees selects split cut-points <b>completely at random</b> from a uniform distribution between the feature's minimum and maximum values in the node. "
        "This extreme randomization reduces model variance, suppresses cut-point optimization bias on continuous GPS coordinates, and provides substantial computational speedups."
    )
    story.append(Paragraph(sec2_text, body_style))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # Section 3: Model Training and Comparative Evaluation
    # ---------------------------------------------------------
    story.append(Paragraph("3. Model Training, Optimization & Comparative Evaluation (20 Marks)", h1_style))
    sec3_text = (
        "<b>3.1 Experimental Setup:</b> The dataset was partitioned into an 80% training set (N = 1,343) and a 20% holdout test set (N = 336) "
        "using stratified sampling across price quintiles. A 5-fold cross-validation protocol was executed on the training partition for all models. "
        "Four models were tuned and benchmarked: (i) Ridge Regression (baseline), (ii) Random Forest (baseline), (iii) Extra Trees (new), and (iv) EBM (new)."
    )
    story.append(Paragraph(sec3_text, body_style))

    # Table 1: Benchmark Results
    table_data = [
        ["Model", "Category", "Syllabus Status", "Test R²", "RMSE ($AUD)", "MAE ($AUD)", "MAPE (%)", "5-Fold CV R²"],
        ["Ridge Regression", "Linear Regularized", "Covered Baseline", "0.6174", "$374,077", "$250,925", "20.72%", "0.6024 ± 0.040"],
        ["Random Forest Regressor", "Bagged Trees", "Covered Baseline", "0.6329", "$366,453", "$256,537", "19.84%", "0.6073 ± 0.020"],
        ["Extra Trees Regressor", "Randomized Ensemble", "New Algorithm 1", "0.6684", "$348,262", "$241,195", "18.46%", "0.6231 ± 0.025"],
        ["Explainable Boosting Machine", "Glass-Box GA²M", "New Algorithm 2", "0.6871", "$338,306", "$233,371", "18.38%", "0.6316 ± 0.020"]
    ]
    t = Table(table_data, colWidths=[1.4*inch, 1.1*inch, 1.1*inch, 0.65*inch, 0.85*inch, 0.85*inch, 0.65*inch, 0.95*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3799")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#d4edda")),  # Highlight EBM
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dcdde1"))
    ]))
    story.append(t)
    story.append(Paragraph("<b>Table 1:</b> Comparative performance benchmark on holdout test set (N = 336 properties in Burwood East & surrounds).", caption_style))
    story.append(Spacer(1, 6))

    if os.path.exists("artifacts/figures/model_benchmark_comparison.png"):
        story.append(Image("artifacts/figures/model_benchmark_comparison.png", width=6.8 * inch, height=2.6 * inch))
        story.append(Paragraph("<b>Figure 5:</b> Test R² Score (higher is better) and Mean Absolute Error (MAE, lower is better) across benchmarked models.", caption_style))

    if os.path.exists("artifacts/figures/predicted_vs_actual_all_models.png"):
        story.append(Image("artifacts/figures/predicted_vs_actual_all_models.png", width=6.4 * inch, height=5.5 * inch))
        story.append(Paragraph("<b>Figure 6:</b> Predicted vs. Actual sale prices ($ Millions AUD) with reference 45-degree parity lines across all 4 models.", caption_style))

    story.append(PageBreak())

    # ---------------------------------------------------------
    # Section 4: Explainability & Interpretation (XAI)
    # ---------------------------------------------------------
    story.append(Paragraph("4. Explainability & Model Interpretation (XAI) (20 Marks)", h1_style))
    sec4_text = (
        "<b>4.1 Global Importance & Feature Hierarchy:</b> EBM calculates the exact mean absolute dollar contribution of each feature. "
        "As seen in Figure 7, distance to Melbourne CBD is the dominant macro driver (|Δ| ≈ $142,000), followed by land size (|Δ| ≈ $98,000), "
        "Mount Waverley Secondary College zone (|Δ| ≈ $85,000), distance to train stations (|Δ| ≈ $62,000), and Burwood One retail hub proximity.<br/><br/>"
        "<b>4.2 Non-Linear Marginal Shape Trajectories:</b> Unlike linear models with fixed slopes, EBM's 1D shape plots (Figure 8) capture realistic economic thresholds: "
        "(i) <i>Land size</i> exhibits rapid price appreciation up to 700 m² before experiencing diminishing marginal returns past 800 m²; "
        "(ii) <i>Train station distance</i> shows a positive walkability premium within 1.2 km, turning into a steep penalty beyond 2.0 km; "
        "(iii) <i>Burwood One proximity</i> reflects a localized convenience premium within 1 km.<br/><br/>"
        "<b>4.3 TreeSHAP Cross-Validation & Local Case Studies:</b> Applying TreeSHAP to the Extra Trees ensemble independently corroborated the feature importance hierarchy (Figure 9). "
        "Local waterfall explanations (Figures 10 & 11) illustrate how the model transparently breaks down individual valuations for buyers and mortgage auditors."
    )
    story.append(Paragraph(sec4_text, body_style))

    if os.path.exists("artifacts/figures/xai_ebm_global_importance.png"):
        story.append(Image("artifacts/figures/xai_ebm_global_importance.png", width=6.2 * inch, height=3.8 * inch))
        story.append(Paragraph("<b>Figure 7:</b> EBM global term importance showing the top 15 main effects and interaction terms ($ AUD valuation impact).", caption_style))

    if os.path.exists("artifacts/figures/xai_ebm_1d_shape_curves.png"):
        story.append(Image("artifacts/figures/xai_ebm_1d_shape_curves.png", width=6.8 * inch, height=4.8 * inch))
        story.append(Paragraph("<b>Figure 8:</b> EBM 1D shape function curves illustrating non-linear valuation effects for CBD distance, land size, train proximity, and Burwood One.", caption_style))

    if os.path.exists("artifacts/figures/xai_shap_summary_beeswarm.png"):
        story.append(Image("artifacts/figures/xai_shap_summary_beeswarm.png", width=6.2 * inch, height=4.2 * inch))
        story.append(Paragraph("<b>Figure 9:</b> TreeSHAP global summary beeswarm plot computed on the Extra Trees ensemble, cross-validating feature attributions.", caption_style))

    if os.path.exists("artifacts/figures/xai_local_case1_mount_waverley.png"):
        story.append(Image("artifacts/figures/xai_local_case1_mount_waverley.png", width=6.2 * inch, height=3.5 * inch))
        story.append(Paragraph("<b>Figure 10:</b> Local instance SHAP waterfall plot for a premium family home in Mount Waverley (MWSC zone).", caption_style))

    story.append(PageBreak())

    # ---------------------------------------------------------
    # Section 5: Critical Discussion
    # ---------------------------------------------------------
    story.append(Paragraph("5. Critical Discussion & Limitations (10 Marks)", h1_style))
    sec5_text = (
        "<b>5.1 Model Limitations:</b> EBM's additive structure restricts interactions to pairwise terms (fᵢⱼ), meaning 3-way or higher-order synergies "
        "(e.g., luxury finish combined with school zone and elevated street topography) cannot be explicitly modeled. Furthermore, tree-based models cannot extrapolate "
        "beyond the maximum land size or price observed in training data.<br/><br/>"
        "<b>5.2 Data Quality & Web Scraping Biases:</b> Web scraping only captures publicly listed properties. Properties traded off-market or through discreet private negotiations "
        "are systematically omitted. In addition, uncaptured latent factors such as architectural aesthetics, natural lighting, and interior renovation quality remain unmodeled in standard tabular listings.<br/><br/>"
        "<b>5.3 Algorithmic Fairness & Ethical Implications:</b> Automated valuation models risk reinforcing historical socioeconomic segregation. Embedding school catchment boundaries "
        "directly into lending algorithms creates positive feedback loops where zoned properties receive preferential mortgage valuations, further locking out lower-income families. "
        "However, glass-box architectures like EBM fulfill Australian regulatory transparency mandates, granting consumers the right to audit and contest automated valuation decisions."
    )
    story.append(Paragraph(sec5_text, body_style))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # Section 6: Front-End Web Application Architecture (Bonus)
    # ---------------------------------------------------------
    story.append(Paragraph("6. Bonus: Front-End Web Application Architecture (+10 Marks)", h1_style))
    sec6_text = (
        "To fulfill the bonus requirement, an interactive dashboard was built using <b>Streamlit</b> (<code>src/app.py</code>). "
        "The system allows non-technical users to:<br/>"
        "• Configure custom property attributes via sliders and selectors (suburb, beds, baths, land size, school zones, distances to transit/shops).<br/>"
        "• View real-time valuation predictions from both EBM ($1.26M) and Extra Trees alongside a 90% confidence interval.<br/>"
        "• Inspect an interactive Plotly horizontal bar/waterfall chart showing the exact dollar attributions calculated by EBM for the specific property.<br/>"
        "• Use a 'What-If' Renovation & Amenity Simulator to evaluate value gains from adding bedrooms, bathrooms, or re-zoning into Mount Waverley Secondary.<br/>"
        "• Test 3 one-click preset profiles (Burwood East family home, Mount Waverley luxury residence, Burwood student unit near Deakin)."
    )
    story.append(Paragraph(sec6_text, body_style))
    story.append(Spacer(1, 8))

    # References
    story.append(Paragraph("References", h2_style))
    refs_text = (
        "1. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. <i>Machine Learning</i>, 63(1), 3-42.<br/>"
        "2. Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. In <i>Proc. 18th ACM SIGKDD</i> (pp. 150-158).<br/>"
        "3. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. In <i>Proc. 19th ACM SIGKDD</i> (pp. 623-631).<br/>"
        "4. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. <i>Nature Machine Intelligence</i>, 2(1), 56-67.<br/>"
        "5. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. <i>arXiv:1909.09223</i>.<br/>"
        "6. Real Estate Institute of Victoria (REIV). (2024). <i>Melbourne School Zone Property Market Insights</i>. Melbourne, Australia."
    )
    story.append(Paragraph(refs_text, caption_style))

    doc.build(story)
    print(f"Academic PDF compiled successfully: {output_pdf_path}")
    return output_pdf_path


if __name__ == "__main__":
    build_academic_pdf()
