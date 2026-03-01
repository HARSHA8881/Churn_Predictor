import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_data(uploaded_file):
    """Phase 1: Load and pre-clean the raw CSV dataset."""
    raw_df = pd.read_csv(uploaded_file)
    df = raw_df.copy()
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    return raw_df, df


def clean_data(df):
    """Phase 2: Impute missing values for numeric and categorical columns."""
    X = df.drop('Exited', axis=1)
    Y = df['Exited']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    expected_cat_features = {'Geography', 'Gender'}
    cat_cols = list(set(cat_cols).union(expected_cat_features).intersection(set(X.columns)))
    num_cols = [f for f in num_cols if f not in cat_cols]

    cat_imputer = SimpleImputer(strategy="most_frequent")
    if len(cat_cols) > 0:
        cat_imputer.fit(X[cat_cols])
        X[cat_cols] = cat_imputer.transform(X[cat_cols])

    num_imputer = SimpleImputer(strategy="mean")
    if len(num_cols) > 0:
        num_imputer.fit(X[num_cols])
        X[num_cols] = num_imputer.transform(X[num_cols])

    return X, Y, num_cols, cat_cols


def encode_features(X, cat_cols):
    """Phase 3: Label-encode categorical columns."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    return X, encoders


def scale_features(X, num_cols):
    """Phase 4: Log-transform then MinMax-scale numeric features."""
    X[num_cols] = np.log1p(X[num_cols])
    minmax = MinMaxScaler()
    columns = X.columns
    X_scaled = minmax.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=columns)
    return X, minmax
