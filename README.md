# Bank Customer Churn Prediction Framework

## 📌 Problem Statement
Customer attrition (churn) is one of the most significant expenses for financial institutions. The cost of acquiring a new customer is substantially higher than retaining an existing one. This project provides a robust, automated classical machine learning framework that predicts the likelihood of a customer leaving the bank based on historical demographic and financial behavior data.

## 🚀 Features
The application is built completely utilizing traditional Machine Learning methodologies and offers 3 distinct sub-features:
1. **Interactive Single-Customer Inference API**: A dynamic prediction engine that processes a single unseen customer's data through the exact imputation and scaling pipeline to deliver live churn risk assessments.
2. **Comprehensive Exploratory Data Analysis (EDA) Dashboard**: Live visualizations of raw data including churn distributions, geographical disparities, and correlation heatmaps to understand feature importance mathematically.
3. **Business Impact & Revenue Calculator**: A dynamic financial simulation that securely translates ML metrics (True Positives, False Positives) into actionable bottom-line dollar figures (Net Profit vs. Loss) based on intervention costs.

## 🧠 Methodology
This project strictly enforces a Scikit-Learn based standard data processing pipeline:
- **Phase 1.** Raw Data Ingestion (Pandas)
- **Phase 2.** Missing Value Imputation (`SimpleImputer` using mean and most frequent)
- **Phase 3.** Categorical Encoding (`LabelEncoder`)
- **Phase 4.** Feature Scaling and Outlier mitigation (`np.log1p` and `MinMaxScaler`)
- **Phase 5.** Model Training (`LogisticRegression` and `DecisionTreeClassifier`)
- **Phase 6.** Serialization (`joblib`)

## 🛠 Tech Stack
- **Frontend & UI**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualizations**: Matplotlib, Seaborn

## 💻 Local Installation
To run this project locally on your machine, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📊 Evaluation & Optimization
The models were evaluated heavily using Cross-Validation, Accuracy, Precision, Recall, and F1-Scores to combat the inherent imbalance of Churn datasets. Optimizations primarily involved proper data scaling and standard continuous logic handling to prevent data leakage and provide better generalization over the decision boundaries.
# Churn_Predictor
