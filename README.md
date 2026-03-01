# Customer Churn Prediction Framework



---

## Problem Statement

Customer attrition (churn) is one of the most significant financial challenges for banking institutions. The cost of acquiring a new customer is 5–7× higher than retaining an existing one. This project builds a robust, end-to-end classical machine learning framework that predicts the probability of a bank customer leaving based on historical demographic and financial behaviour data — enabling targeted, data-driven retention strategies.

---

## Live Demo

**[Click here to open the live app](https://churnpredictorgenai.streamlit.app/)**

---

## Team

| Name | Role |
|------|------|
| Harsha Gonela (2401010181) | Model Development, Streamlit UI, Deployment |
| Shivansh Upadhyaya (2401020109) | Data Cleaning & EDA |
| Gourav Tanwar (2401010173) | Model Development, Streamlit UI, Deployment |

---

## Dataset

- **Source**: [Kaggle — Bank Customer Churn Modelling](https://mavenanalytics.io/data-playground/bank-customer-churn)
- **Size**: 10,000 rows × 14 features
- **Target**: `Exited` (1 = churned, 0 = retained) — ~20% positive class

### Key Features
| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer credit score (300–850) |
| Geography | Categorical | France / Germany / Spain |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years with the bank |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products used |
| HasCrCard | Binary | Has credit card (1/0) |
| IsActiveMember | Binary | Active member status (1/0) |
| EstimatedSalary | Numeric | Annual estimated salary |

---

## Project Structure

```
Customer churn prediction/
├── app.py                      ← Home dashboard
├── model_train.py              ← Standalone CLI training script
├── requirements.txt
├── .gitignore
│
├── pages/
│   ├── 1_About.py              ← Project info and methodology
│   ├── 2_Dataset_Explorer.py   ← CSV upload + EDA visualisations
│   ├── 3_Model_Training.py     ← One-click model training
│   ├── 4_Performance.py        ← Metrics, charts, business impact
│   └── 5_Churn_Predictor.py    ← Live single-customer prediction
│
├── src/
│   ├── model.py                ← Training, evaluation, save/load
│   └── styles.py               ← Shared CSS for all pages
│
├── utils/
│   └── data_utils.py           ← Data loading and preprocessing
│
├── models/                     ← Saved .pkl artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── minmax_scaler.pkl
│
└── data/                       ← Place your CSV dataset here
```

---

## Sub-Features (Technical Depth)

### 1. Custom 6-Phase ML Data Pipeline
A structured, modular preprocessing pipeline implemented in `utils/data_utils.py`:
- **Phase 1** — Raw data ingestion via Pandas; drops non-predictive ID columns
- **Phase 2** — Missing value imputation (`SimpleImputer` — mean for numerics, most-frequent for categoricals)
- **Phase 3** — Categorical encoding (`LabelEncoder` on Geography and Gender)
- **Phase 4** — Outlier mitigation via `np.log1p` followed by `MinMaxScaler` normalization to [0, 1]
- **Phase 5** — Stratified 70/30 train-test split to preserve class ratio
- **Phase 6** — Model serialization via `joblib` to `.pkl` artifacts in `models/`

### 2. Multi-Model Comparative Training Engine
`src/model.py` trains and evaluates 4 classifiers simultaneously:
- **Logistic Regression** — `class_weight='balanced'`, `C=0.5`, 2000 iterations
- **Decision Tree** — `class_weight='balanced'`, `max_depth=10`, pruned leaves
- **Random Forest** — 300 trees, `class_weight='balanced'`, `max_depth=15`, parallel (`n_jobs=-1`)
- **Gradient Boosting** — 200 rounds, `learning_rate=0.05`, `subsample=0.8`

All models use `class_weight='balanced'` or equivalent to address the inherent 80/20 class imbalance in churn datasets.

### 3. Interactive EDA Dashboard
`pages/2_Dataset_Explorer.py` provides live, upload-driven visualizations:
- Churn distribution pie chart
- Countplot by geography
- Age and Balance distribution histograms (by churn label)
- Full Pearson Correlation Heatmap

### 4. Business Impact Calculator
`pages/4_Performance.py` (Business Impact tab) translates ML metrics into financial figures:
- Revenue Saved = `True Positives × Cost per Churned Customer`
- Net Profit = Revenue Saved − `(TP + FP) × Intervention Cost`
- Adjustable costs via Streamlit number inputs

### 5. Real-Time Single-Customer Inference Engine
`pages/5_Churn_Predictor.py` applies the full saved pipeline to new data:
- Re-applies saved `LabelEncoder` objects for categorical features
- Applies `log1p` transformation followed by the saved `MinMaxScaler`
- Outputs probability scores from all 4 models
- Ensemble average with vote count for final verdict

---

## Methodology & Algorithm Justification

### Why 4 Models?
Rather than committing to a single algorithm, we train a suite to compare and select the best:

| Model | Justification |
|-------|--------------|
| Logistic Regression | Linear baseline; interpretable; fast inference |
| Decision Tree | Non-linear boundaries; visual explainability; feature importance |
| Random Forest | Ensemble averaging reduces variance of single DT dramatically |
| Gradient Boosting | Sequential error correction; consistently highest accuracy on tabular data |

### Why `class_weight='balanced'`?
The Churn dataset has ~20% positive class. Without correction, models achieve 80% accuracy by simply predicting "stays" for every customer — a meaningless result. `class_weight='balanced'` weights minority class examples by `total_samples / (2 × positive_samples)`, forcing models to genuinely learn churn patterns.

### Why `log1p` + `MinMaxScaler`?
Features like `Balance` and `EstimatedSalary` have heavy right-skewed distributions with outliers. `log1p` compresses the long tail, making the distribution more Gaussian-like before scaling. `MinMaxScaler` then normalizes to [0, 1] for Logistic Regression convergence.

---

## Evaluation Results

> Train models using the Dataset Explorer page, then check the Performance page for live results.

Metrics computed on the held-out 30% test set:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted churners, how many actually churned |
| Recall | Of actual churners, how many were caught |
| F1 Score | Harmonic mean of Precision and Recall |
| AUC-ROC | Area under the ROC curve (model discrimination power) |
| CV Mean F1 | 5-fold cross-validated F1 on training set |

---

## Optimization Steps

1. **Stratified split** — ensures equal churn ratio in train and test sets
2. **`class_weight='balanced'`** — corrects class imbalance without oversampling
3. **Hyperparameter tuning** — `max_depth`, `min_samples_leaf`, `learning_rate`, `subsample` tuned per model
4. **Gradient Boosting `subsample=0.8`** — stochastic boosting reduces overfitting
5. **Random Forest `n_estimators=300`** — sufficient trees for variance stabilization
6. **`log1p` transform** — suppresses outlier effect on `Balance` and `EstimatedSalary`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI & App | Streamlit |
| ML Models | Scikit-Learn (LR, DT, RF, GB) |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Persistence | joblib (.pkl) |
| Version Control | Git + GitHub |

---

## Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/HARSHA8881/Churn_Predictor.git
cd Churn_Predictor

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## System Architecture

```
[User uploads CSV]
        |
        v
[Data Loading] ─► Drop ID columns (RowNumber, CustomerId, Surname)
        |
        v
[Imputation] ─► Mean (numerics) | Most-Frequent (categoricals)
        |
        v
[Encoding] ─► LabelEncoder (Geography, Gender)
        |
        v
[Scaling] ─► log1p transform ─► MinMaxScaler [0,1]
        |
        v
[Training] ─► LR | DT | RF | GB (stratified 70/30 split)
        |
        v
[Serialization] ─► logistic_regression.pkl | decision_tree.pkl
                    random_forest.pkl | gradient_boosting.pkl
                    minmax_scaler.pkl
        |
        v
[Prediction] ─► Re-apply encoders ─► log1p ─► MinMaxScaler ─► predict_proba()
        |
        v
[Output] ─► 4 model scores + ensemble average + business impact
```

---

## Viva Voce Preparation Notes

**Q: Why not use Neural Networks?**
A: For a tabular dataset of 10K rows with 10 features, ensemble methods like Random Forest and Gradient Boosting consistently outperform neural networks. Neural networks require significantly more data and hyperparameter engineering to surpass tree-based ensembles on structured/tabular data.

**Q: How does data flow from input to output?**
A: CSV → ID column removal → imputation → label encoding → log1p + MinMaxScaler → train/test split → model.fit() → joblib serialization. For live prediction: user input → DataFrame → same encoders + scaler applied → predict_proba() → ensemble average.

**Q: Why MinMaxScaler and not StandardScaler?**
A: After `log1p`, the distributions are approximately bounded and less Gaussian than raw data. MinMaxScaler guarantees [0,1] range for all features, which is beneficial for both Logistic Regression convergence and tree-based model consistency.
