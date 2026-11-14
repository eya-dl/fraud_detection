# Fraud Detection on Mobile-Money Transactions

> Detect fraudulent transfers among **6.3 million** mobile-money transactions (PaySim), compare supervised and unsupervised models under extreme class imbalance, track every experiment with **MLflow**, and explore the results in an analyst dashboard.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-EB5E28)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)

## Key results

Fraud is **0.13 %** of transactions and only appears in `TRANSFER` and `CASH_OUT`, so accuracy is meaningless: models are ranked by **PR-AUC** on a held-out future period.

| Model | Imbalance strategy | PR-AUC | ROC-AUC |
|---|---|---|---|
| **XGBoost** | class weighting | **0.917** | 0.998 |
| XGBoost | none | 0.917 | 0.999 |
| XGBoost | SMOTENC | 0.896 | 0.998 |
| Random Forest | none | 0.870 | 0.995 |
| Logistic Regression | SMOTENC | 0.690 | 0.980 |

**Business reading** — if analysts can review only the top **0.5 %** most suspicious transactions, the best model catches **91 %** of frauds; at the top 0.1 %, every alert is a real fraud (precision 1.0). The tuned model also flags transactions covering **99 %** of the fraudulent amount.

## What makes it rigorous

- **Temporal split** (train → validation → test by time step) instead of a random split, so the model is always evaluated on the future, as in production.
- **Three imbalance strategies** compared on the same pipelines: none, class weighting, SMOTENC (oversampling that handles categorical features).
- **Decision threshold tuned on validation**, then reported on test, plus an **alert-budget** view (recall at top 0.1 / 0.5 / 1 % of scores).
- **Unsupervised baselines** (Isolation Forest, Local Outlier Factor, One-Class SVM) with a contamination sweep, for the case where labels are not available.
- **Every run logged to MLflow** (parameters, metrics, confusion matrices, PR curves, models), with a tagging contract that the dashboard reads.

## Dashboard

`fraud_dashboard.py` is a Streamlit app that reads the MLflow runs directly: overview, model comparison, PR/ROC curves, and a threshold-vs-cost simulator (false alarms vs missed frauds).

## Project structure

```
machine_learning/
├── eda_featureeng.ipynb              exploration + feature engineering
├── supervised_fraud_detection.ipynb  LR / RF / XGBoost × 3 imbalance strategies, MLflow
├── unsupervised_fraud_detection.ipynb Isolation Forest / LOF / One-Class SVM, MLflow
├── fraud_dashboard.py                Streamlit dashboard on top of MLflow
├── models/                           saved pipelines (baseline, weighted, smote)
├── mlflow_artifacts/                 confusion matrices and PR curves per run
└── visualizations/                   comparison charts
```

## Getting started

```bash
git clone https://github.com/eya-dl/fraud_detection.git
cd fraud_detection
pip install -r requirements.txt
```

1. Download the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and save it as `machine_learning/data.csv`.
2. Run `eda_featureeng.ipynb` first: it writes the engineered features to `data/processed/fraud_eda_features.parquet`.
3. Run the supervised and unsupervised notebooks. Runs are written to `machine_learning/mlflow.db`.
4. Explore:
   ```bash
   cd machine_learning
   mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking
   streamlit run fraud_dashboard.py                    # analyst dashboard
   ```

## Tech stack

pandas · scikit-learn · imbalanced-learn · XGBoost · MLflow · Streamlit · Plotly

## Author

**Eya Dhambri** — AI engineering student at ENSI (Tunisia) · [LinkedIn](https://www.linkedin.com/in/eya-dhambri) · eya.dhambri@ensi-uma.tn
