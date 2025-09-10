

# Predicting County-Level Diabetes Prevalence in the U.S. (2016–2019)

> Machine Learning & Geospatial Analysis for Public Health  
> **Author**: Joseph Grob | MSc Student in Geography, University of Lausanne  
> **Date**: December 2024  

---

## Project Overview

This project aims to predict the prevalence of diabetes at the **county level** in the United States using **machine learning models** and **geospatial analysis**. It explores how lifestyle, demographic, environmental, and socioeconomic factors contribute to diabetes, with a strong emphasis on regional variation and interpretability.

The analysis focuses on **2019 predictions**, using data from **2016 to 2018** for training.

---

## Objectives

- ✅ Predict the **percentage of people diagnosed with diabetes** at the U.S. county level.
- ✅ Identify and rank the **key factors** contributing to diabetes prevalence.
- ✅ Analyze **regional disparities** using geospatial and clustering techniques.
- ✅ Provide an **interpretable and reproducible** ML pipeline.

---

## Models Used

Several models were implemented and compared based on performance (R², RMSE):

| Model | Description | Final Use |
|-------|-------------|-----------|
| **Linear Regression (Baseline)** | Simple model for comparison | ❌ |
| **Random Forest** | Non-linear ensemble model | ❌ |
| **Geographical Random Forest** | RF + latitude/longitude as predictors | ❌ |
| **XGBoost** | Gradient boosting model with SHAP analysis | ✅ **Selected** |

The final model, **XGBoost**, was chosen for its balance of **predictive power** and **interpretability**, with:
- `R² test = 0.4158`
- `RMSE test = 0.7643`

---

## Technical Highlights

### Libraries used
- `pandas` / `numpy` – Data cleaning and manipulation  
- `matplotlib` / `seaborn` – Visualization  
- `scikit-learn` – Linear regression, preprocessing, model selection  
- `xgboost` – Final machine learning model  
- `shap` – Model interpretability with SHAP values  
- `kmeans` from `scikit-learn` – Clustering counties  
- `geopandas` / `plotly` / `folium` – Spatial mapping of results  


---

## Model Interpretation (SHAP)

SHAP (SHapley Additive exPlanations) was used to understand which features most influence the predictions.

**Top 3 influential features across counties**:
1. `% Physically Inactive`
2. `% Excessive Drinking`
3. `% Obesity`

The dominant factor varies by region.  
Urban areas show more diverse influences; rural and southeastern U.S. counties cluster around inactivity, smoking, and obesity.

---

## Data

The dataset includes U.S. county-level data from **2016 to 2019**, including:
- Demographics
- Socioeconomics
- Health behaviors
- Environmental data

🗂️ **Dataset Repository** (open access):  
🔗 [https://github.com/JosephGrob/DATASET_FINAL_PROJECT_MLEES_2024](https://github.com/JosephGrob/DATASET_FINAL_PROJECT_MLEES_2024)

All files (`.csv`, `.xlsx`, including `lat/lon`) can be found and downloaded there.

---

## Methodology Summary

1. **Preprocessing**: Merge, clean, and normalize multi-year data  
2. **Feature Selection**: OLS + AIC/BIC to reduce dimensionality  
3. **Model Training**: Compare multiple models on 2016–2018 data  
4. **Model Evaluation**: Predict 2019, assess performance  
5. **Interpretability**: SHAP to rank feature importance  
6. **Spatial Analysis**: Cluster counties based on SHAP profiles

---

## Conclusion

- **Physical inactivity**, **excessive drinking**, and **obesity** are consistently the strongest predictors of diabetes.
- Regional clustering reveals strong disparities, especially in the Southeast ("Diabetes Belt").
- Though XGBoost outperformed other models, the R² remains moderate, suggesting more complex or unmeasured factors influence diabetes prevalence.

---

## Future Work

- Integrate **additional geospatial features** (e.g., walkability, food access)
- Use **temporal models** (e.g., LSTM) to capture time trends
- Apply the same pipeline to other chronic diseases (e.g., obesity, heart disease)

---

## 📫 Contact

**Joseph Grob**  
Spatial Analyst Student (MSc) – University of Lausanne  
📍 Lausanne, Switzerland  
✉️ grob.j1890@gmail.com  
📎 GitHub: [@JosephGrob](https://github.com/JosephGrob)

---


==================== .gitignore ====================

# Ignore Jupyter notebook checkpoints
.ipynb_checkpoints/

# Ignore Python cache files
__pycache__/
*.py[cod]

# Ignore data files if needed
*.csv
*.xlsx

# Ignore system files
.DS_Store
Thumbs.db

# VS Code settings
.vscode/

# Environments
.env
.venv
env/
venv/

==================== requirements.txt ====================

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap
geopandas
plotly
folium
openpyxl


