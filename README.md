# 🏠 House Prices — Advanced Regression Techniques

> Kaggle competition solution achieving **~0.111 RMSE** (Top 10%)  
> End-to-end ML pipeline with stacking ensemble: Ridge · Lasso · XGBoost · LightGBM · CatBoost

---

## 📌 Competition

**[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**

- **Task:** Predict residential home sale prices in Ames, Iowa
- **Dataset:** 79 explanatory variables describing (almost) every aspect of residential homes
- **Metric:** RMSLE — Root Mean Squared Log Error

---

## 📁 Repository Structure

```
house-prices-ensemble/
│
├── house_prices_kaggle.ipynb   # Main pipeline notebook (Kaggle-ready)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🔧 Pipeline Overview

```
Raw Data (1460 rows, 79 features)
        │
        ├── 1. Outlier Removal       (2 rows only — GrLivArea anomalies)
        ├── 2. Missing Value Strategy (semantic NA → "None", LotFrontage → neighborhood median)
        ├── 3. Ordinal Encoding       (quality/condition columns → ranked integers)
        ├── 4. Feature Engineering    (30+ new features)
        │     ├── Area aggregations   (TotalSF, TotalBathrooms, TotalPorchSF)
        │     ├── Time features       (HouseAge, IsRemodeled, YearsSinceRemod)
        │     ├── Quality × Area      (QualArea, BsmtScore, KitchenScore)
        │     ├── Polynomial terms    (OverallQual², OverallQual³)
        │     └── Boolean flags       (HasPool, HasGarage, HasFireplace...)
        ├── 5. One-Hot Encoding       (nominal categoricals)
        ├── 6. Box-Cox Transform      (skewed continuous features, |skew| > 0.5)
        └── 7. Variance Filter        (remove near-zero variance features)
                │
                ▼
        Level-1 Models (10-Fold OOF)
        ├── Ridge          OOF RMSE: 0.11005
        ├── Lasso          OOF RMSE: 0.11018
        ├── ElasticNet     OOF RMSE: 0.11022
        ├── CatBoost       OOF RMSE: 0.11391
        ├── XGBoost        OOF RMSE: 0.11569
        └── LightGBM       OOF RMSE: 0.11793
                │
                ▼
        Level-2 Ensemble
        ├── Stacking  (RidgeCV meta-model)  OOF RMSE: 0.10843
        ├── Blending  (scipy-optimized weights) OOF RMSE: 0.10830
        └── Final     (10% Stack + 90% Blend)   OOF RMSE: 0.10830
```

---

## 🚀 Quick Start

### Local

```bash
# 1. Clone the repo
git clone https://github.com/goktani/house-prices-ensemble.git
cd house-prices-ensemble

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from Kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip

# 4. Run the notebook
jupyter notebook house_prices_kaggle.ipynb
```

### Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **New Notebook → File → Import Notebook**
3. Upload `house_prices_kaggle.ipynb`
4. Add the competition dataset from the **Data** panel
5. Click **Run All**

---

## 📊 Results

| Model | OOF RMSE |
|-------|----------|
| Ridge | 0.11005 |
| Lasso | 0.11018 |
| ElasticNet | 0.11022 |
| CatBoost | 0.11391 |
| XGBoost | 0.11569 |
| LightGBM | 0.11793 |
| **Stacking Ensemble** | **0.10843** |
| **Blending Ensemble** | **0.10830** |
| **Final (Blend 90% + Stack 10%)** | **0.10830** |

---

## 💡 Key Design Decisions

**Outlier Removal — Why only 2 rows?**  
The competition host explicitly recommends removing only the 2 houses with `GrLivArea > 4000` that sold abnormally cheap. Removing more rows causes the model to never see high-value homes (`$300k+`), capping predictions at ~$280k.

**Ordinal vs One-Hot Encoding**  
Quality/condition columns (`ExterQual`, `KitchenQual`, etc.) are ordinal — `Ex > Gd > TA > Fa > Po`. Encoding them as ranked integers preserves this hierarchy, especially beneficial for linear models.

**Why Blending Dominates (α=0.90)?**  
Ridge/Lasso have very strong OOF scores (0.110) and the scipy-optimized blending assigns them ~65% combined weight. The stacking meta-model adds marginal improvement, but simple weighted blending is more robust here.

**LotFrontage Strategy**  
Filled with neighborhood median rather than global median — homes on the same street tend to have similar frontage widths.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.4.2 | Linear models, CV, preprocessing |
| XGBoost | 2.0.3 | Gradient boosting |
| LightGBM | 4.3.0 | Gradient boosting (fast) |
| CatBoost | 1.2.5 | Gradient boosting (ordered) |
| Optuna | 3.6.1 | Hyperparameter optimization |
| SciPy | 1.13.0 | Box-Cox transform, blend optimization |
| Pandas | 2.2.2 | Data manipulation |

---

## 📜 License

MIT License — feel free to use, modify, and distribute.
