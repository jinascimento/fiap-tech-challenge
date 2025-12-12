import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report
)
from sklearn.neural_network import MLPClassifier


def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    irq = q3 - q1
    return df[(df[column] >= q1 - 1.5 * irq) & (df[column] <= q3 + 1.5 * irq)]

def prepare_dataset(df):
    # remove duplicates
    df.drop_duplicates(inplace=True)

    # remove outliers
    df = remove_outliers(df, 'bmi')
    df = remove_outliers(df, 'blood_glucose_level')

    # remove unnecessary data
    df_cleaned = df.copy()
    df_cleaned = df_cleaned[df_cleaned['gender'] != 'Other']

    bins = [0, 12, 19, 39, 59, 81]
    labels = ['Criança', 'Adolescente', 'Jovem', 'Adulto', 'Idoso']
    df_cleaned['age_bins'] = pd.cut(df_cleaned['age'], bins=bins, labels=labels, right=False)
    df_cleaned = df_cleaned.drop(columns=['age', 'bmi_category'], errors='ignore')
    df_processed = df_cleaned.copy()
    df_processed = pd.get_dummies(df_processed, columns=['gender', 'smoking_history', 'age_bins'], drop_first=True)
    return df_processed


def load_dataset():
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "..", "data", "diabetes_prediction_dataset.csv")

    return pd.read_csv(dataset_path)

def build_mlp_baseline(df_processed):
    X = df_processed.drop('diabetes', axis=1)
    y = df_processed['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=42,
    )

    mlp_model.fit(X_train, y_train)
    return mlp_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred_mlp = model.predict(X_test)
    report_mlp = classification_report(y_test, y_pred_mlp)

    return report_mlp