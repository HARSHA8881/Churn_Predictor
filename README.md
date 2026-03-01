# CUSTOMER CHURN PREDICTION FRAMEWORK

## Project Overview

This project involves the design and implementation of a system that evaluates the probability of a bank customer churning (leaving) and supports data-driven retention decisions. It is a machine learning-powered churn analytics system built on a multi-page Streamlit interface, enabling financial institutions to identify at-risk customers before they leave.

---

## Problem Statement

Financial institutions face significant challenges in retaining customers. Manual identification of at-risk customers is time-consuming, inconsistent, and reactive — action is often taken only after the customer has already left.

This project addresses this problem by implementing an automated customer churn scoring system that uses machine learning algorithms to analyze borrower demographic and financial data, classify customers by churn likelihood, and support proactive retention strategies.

---

## Live Demo

**[Click here to open the live app](https://churnpredictorgenai.streamlit.app/)**

---

## Key Features

- Upload customer dataset through an interactive UI
- Automatic 6-phase data preprocessing pipeline
- Support for categorical encoding and feature scaling
- Training and comparison of multiple ML models simultaneously
- Real-time churn risk prediction for individual customers
- Visualization of evaluation metrics, ROC curves, and feature importance
- Business Impact Calculator — translates ML metrics into revenue figures
- Clean and user-friendly multi-page Streamlit interface

---

## Machine Learning Models Used

The following supervised learning models were implemented:

**Logistic Regression**
- Used for probabilistic churn classification
- Estimates probability of a customer leaving
- `class_weight='balanced'` applied to handle class imbalance

**Decision Tree Classifier**
- Rule-based classification model
- Identifies key risk-driving features through splits
- `max_depth=10` and leaf pruning to prevent overfitting

**Random Forest Classifier**
- Ensemble of 300 decision trees
- Reduces variance and overfitting compared to a single tree
- Parallel training with `n_jobs=-1`

**Gradient Boosting Classifier**
- Sequential boosting — each round corrects previous errors
- 200 estimators with `learning_rate=0.05` and `subsample=0.8`
- Consistently highest accuracy on tabular churn data

---

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy Score
- Precision, Recall, and F1 Score
- Cross-Validated F1 (5-fold)
- ROC-AUC Score
- Confusion Matrix
- ROC Curve Visualization
- Feature Importance Analysis (Decision Tree and Random Forest)

---

## Installation and Setup Instructions

Follow these steps to run the project locally.

**Step 1: Clone the Repository**
```bash
git clone https://github.com/HARSHA8881/Churn_Predictor.git
cd Churn_Predictor
```

**Step 2: Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Launch the Application**

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Dataset

- **Source**: [Maven Analytics — Bank Customer Churn](https://mavenanalytics.io/data-playground/bank-customer-churn)
- **Size**: 10,000 rows × 14 features
- **Target Variable**: `Exited` (1 = churned, 0 = retained) — ~20% positive class

---

## Team Contribution

| Member | Contribution |
|--------|-------------|
| Harsha Gonela (2401010181) | Model Development, Streamlit UI, Deployment |
| Shivansh Upadhyaya (2401020109) | Data Cleaning & EDA |
| Gourav Tanwar (2401010173) | Model Development, Streamlit UI, Deployment |

---

## Conclusion

The Customer Churn Prediction Framework successfully demonstrates how Machine Learning can automate the identification of at-risk bank customers. The trained models — particularly Random Forest and Gradient Boosting — achieved strong performance across accuracy, F1, and AUC metrics. The system can assist financial institutions in making proactive, data-driven retention decisions, ultimately reducing customer acquisition costs and increasing long-term profitability.
