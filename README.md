# Retail Store Demand Forecasting
### Hybrid LSTM + XGBoost Ensemble with Explainable AI

---

## Overview

A production-grade machine learning pipeline for daily retail demand forecasting. The system combines a gradient-boosted tree model (XGBoost) with a sequential deep learning model (LSTM) using a Ridge stacking meta-learner to produce a hybrid ensemble forecast.

The model predicts **Units Sold** for any given date range using historical demand patterns, pricing signals, weather conditions, seasonal factors, and calendar features.

---

## Dataset

| Property | Value |
|---|---|
| File | `retail_store_inventory.csv` |
| Rows | 73,100 |
| Date range | 2022-01-01 → 2024-01-01 |
| Stores | 5 (S001–S005) |
| Products | 20 (PROD-001–PROD-020) |
| Categories | Drinks, Confectionery, Snacks, Bakery & Bread, Dairy & Eggs, Grocery |

---

## Project Structure

```
retail-demand-forecasting/
│
├── retail_demand_forecasting.ipynb   # Main notebook (training + forecasting)
├── retail_store_inventory.csv        # Dataset
├── README.md                         # This file
│
├── saved_models/                     # Generated after running the notebook
│   ├── lstm_model.keras
│   ├── xgb_model.pkl
│   ├── meta_learner.pkl
│   ├── lstm_feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── le_store.pkl
│   ├── le_product.pkl
│   ├── ohe_encoder.pkl
│   └── config.pkl
│
└── outputs/                          # Generated plots and forecast CSVs
    ├── eda_overview.png
    ├── eda_business_factors.png
    ├── store_trends.png
    ├── correlation_heatmap.png
    ├── feature_selection.png
    ├── xgb_feature_importance.png
    ├── lstm_training_history.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── scatter_actual_vs_predicted.png
    ├── shap_summary.png
    ├── shap_bar.png
    ├── shap_waterfall.png
    └── forecast_output.png
```

---

## Architecture

```
Raw Data
    │
    ▼
Preprocessing
    │  Remove leaking columns (Demand Forecast, Units Ordered)
    │  Remove identifiers (Product Name)
    │  Cap outliers (IQR × 3)
    │  OneHotEncode: Weather, Region, Category, Seasonality
    │
    ▼
Feature Engineering
    │  Lag features: 1, 2, 3, 7, 14 days
    │  Rolling mean + std: 3, 7, 14, 28 days
    │  Rolling max/min: 7 days
    │  Price × Discount interaction
    │  Effective price, Price vs Competitor, Inventory ratio
    │  Cyclical datetime encoding (sin/cos)
    │
    ▼
Feature Selection  (Mutual Information + F-regression + Lasso voting)
    │
    ▼
Per-Entity Chronological Split  (70% train / 15% val / 15% test)
    │
    ├─────────────────────┬────────────────────────┐
    ▼                     ▼                        ▼
XGBoost               LSTM                   Baselines
(raw features)        (scaled sequences)     (Naive, LR)
    │                     │
    └──────────┬───────────┘
               ▼
    Ridge Stacking Meta-Learner
    (trained on validation predictions)
               │
               ▼
    Hybrid Ensemble Forecast
               │
               ▼
    SHAP Explainability (XGBoost component)
```

---

## Running the Notebook

### Google Colab (recommended)

1. Upload `retail_demand_forecasting.ipynb` and `retail_store_inventory.csv` to Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom
4. After Section 15, all model files download automatically to your PC

### Local environment

```bash
pip install xgboost shap scikit-learn tensorflow pandas numpy matplotlib seaborn joblib

# Place retail_store_inventory.csv in the same directory as the notebook
# Update FILE_PATH in Section 2 if needed
jupyter notebook retail_demand_forecasting.ipynb
```

---

## How to Configure a Forecast

In **Section 14**, set these variables:

```python
FROM_DATE    = '2023-11-01'   # start of forecast period
TO_DATE      = '2023-11-30'   # end of forecast period

PRICE        = 0.85           # selling price (£)
DISCOUNT     = 10             # discount percentage (0–20)
INVENTORY    = 300            # current stock level

WEATHER      = 'Rainy'        # Sunny / Rainy / Cloudy / Snowy
REGION       = 'North'        # North / South / East / West
CATEGORY     = 'Drinks'       # Drinks / Snacks / Confectionery /
                               # Bakery & Bread / Dairy & Eggs / Grocery
SEASONALITY  = 'Autumn'       # Spring / Summer / Autumn / Winter
HOLIDAY      = 0              # 0 = no promotion, 1 = promotion active
```

The forecast output is a table with daily predictions from all three models (LSTM, XGBoost, Ensemble) plus a chart with weekend shading. Results are also saved as a CSV.

**Recommended date range:** Within 2022-01-01 to 2024-01-01 for highest accuracy. Forecasting up to 14 days beyond the data end is acceptable. Beyond that, errors compound through the recursive lag chain.

---

## Evaluation Metrics Explained

| Metric | What it measures | Good value |
|---|---|---|
| **MAE** | Average absolute error in units | Lower is better |
| **RMSE** | Penalises large errors more than MAE | Lower is better |
| **R²** | Proportion of demand variance explained | Closer to 1.0 |
| **MAPE %** | Mean absolute percentage error | Lower is better |
| **SMAPE %** | Symmetric MAPE — stable near zero sales | Lower is better |
| **Accuracy %** | 1 − WMAPE, scaled 0–100% | Higher is better |

**Why R² may be moderate (~0.3–0.5):**
This dataset has high natural variance — 5 very different stores selling 20 very different products in the same feature space. A model trained without Store ID and Product ID (intentionally excluded to avoid memorising identifiers) faces a genuinely hard generalisation problem. The baselines (Naive, Linear Regression) provide the honest floor.

---

## Using Saved Models in a Flask App

```python
# app.py
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load once at startup
lstm_model       = tf.keras.models.load_model('saved_models/lstm_model.keras')
xgb_model        = joblib.load('saved_models/xgb_model.pkl')
meta_learner     = joblib.load('saved_models/meta_learner.pkl')
lstm_feat_scaler = joblib.load('saved_models/lstm_feature_scaler.pkl')
target_scaler    = joblib.load('saved_models/target_scaler.pkl')
ohe              = joblib.load('saved_models/ohe_encoder.pkl')
config           = joblib.load('saved_models/config.pkl')

SELECTED_FEATURES = config['SELECTED_FEATURES']
SEQ_LEN           = config['SEQ_LEN']

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    # Call forecast_date_range() with data['from_date'], data['to_date'], etc.
    # Return forecast_df as JSON
    result = forecast_date_range(
        from_date         = data['from_date'],
        to_date           = data['to_date'],
        price             = data.get('price', 1.0),
        discount          = data.get('discount', 0),
        inventory_level   = data.get('inventory', 200),
        weather           = data.get('weather', 'Sunny'),
        region            = data.get('region', 'North'),
        category          = data.get('category', 'Drinks'),
        seasonality       = data.get('seasonality', 'Summer'),
        holiday_promotion = data.get('holiday', 0),
    )
    result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
    return jsonify(result[['Date','Day','Ensemble_Forecast']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
```

**Example API call:**
```bash
curl -X POST http://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "from_date": "2023-11-01",
    "to_date":   "2023-11-07",
    "price": 0.85,
    "discount": 10,
    "inventory": 300,
    "weather": "Rainy",
    "region": "North",
    "category": "Drinks",
    "seasonality": "Autumn",
    "holiday": 0
  }'
```

---

## Key Design Decisions

**Why Store ID and Product ID are excluded from features:**
These are identifier columns, not demand-driving signals. Including them causes the model to memorise entity-level averages rather than learning transferable demand patterns. A model trained with IDs cannot generalise to new stores or products.

**Why Demand Forecast and Units Ordered are removed:**
`Demand Forecast` is a pre-computed prediction of the target variable — using it is direct target leakage. `Units Ordered` is typically determined after forecasting demand, so it encodes future information.

**Why XGBoost uses raw unscaled features:**
Tree-based algorithms split on feature thresholds. Scaling `x` to `(x - min) / (max - min)` does not change the ordering of values, so the splits are identical. Scaling is unnecessary and adds a transformation dependency.

**Why sequences are built per Store+Product:**
LSTM assumes a continuous temporal sequence. Building sequences globally would create windows that span the boundary between Store A Product 1 and Store B Product 2 — the model would try to learn a temporal relationship between two unrelated time series.

---

## Dependencies

```
tensorflow >= 2.13
xgboost >= 1.7
scikit-learn >= 1.3
shap >= 0.42
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
joblib >= 1.3
```

---

## Licence

For academic and research use.
