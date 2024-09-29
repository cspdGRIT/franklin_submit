import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # Requires xgboost package
from lightgbm import LGBMClassifier  # Requires lightgbm package
from catboost import CatBoostClassifier  # Requires catboost package
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the datasets
loan_data = pd.read_csv('MPLCaseStudy.csv', low_memory=False)
mapping = pd.read_csv('mapping.csv')

# Apply column renaming from mapping.csv
mapping_dict = dict(zip(mapping['LoanStatNew'], mapping['Description']))
loan_data.rename(columns=mapping_dict, inplace=True)

# Function to convert percentage strings to floats
def convert_percent_to_float(x):
    if isinstance(x, str) and '%' in x:
        return float(x.strip('%')) / 100
    return x

# Convert date columns to datetime format and create new features
date_columns = ['The month the borrower\'s earliest reported credit line was opened',
                'The month which the loan was funded']
for col in date_columns:
    loan_data[col] = pd.to_datetime(loan_data[col], errors='coerce')

loan_data['years_since_earliest_cr_line'] = loan_data['The month the borrower\'s earliest reported credit line was opened'].apply(
    lambda x: (pd.Timestamp.now() - x).days / 365 if pd.notnull(x) else None)

# Drop original date columns
loan_data.drop(columns=date_columns, inplace=True)

# Convert percentage columns to float values
percentage_columns = [
    'Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.',
    'Percentage of all bankcard accounts > 75% of limit.',
    'Balance to credit limit on all trades'
]
for col in percentage_columns:
    loan_data[col] = loan_data[col].apply(convert_percent_to_float)

# Handle missing 'loan_status' column
if 'loan_status' not in loan_data.columns:
    loan_data.rename(columns={'Current status of the loan': 'loan_status'}, inplace=True)

# Identify numeric columns that need imputation and remove columns with all missing values
numeric_columns = loan_data.select_dtypes(include=[np.number]).columns.tolist()
loan_data_numeric = loan_data[numeric_columns].dropna(axis=1, how='all')

# Impute missing values for numeric columns using SimpleImputer
imputer = SimpleImputer(strategy='median')
loan_data_imputed = pd.DataFrame(imputer.fit_transform(loan_data_numeric), columns=loan_data_numeric.columns)
loan_data.update(loan_data_imputed)

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in loan_data.select_dtypes(include=[object]):
    label_encoders[column] = LabelEncoder()
    loan_data[column] = label_encoders[column].fit_transform(loan_data[column].astype(str))

# Split data into features and target variable, then into train and test sets
X = loan_data.drop(columns=['loan_status'])
y = loan_data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a list of models to compare (including new models)
models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=500),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'GradientBoosting': GradientBoostingClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),  # Suppress warning for XGBoost 
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True)  # Suppress output for CatBoost training 
}

def evaluate_model_with_timeout(model, X_train, X_test, y_train, y_test, timeout=60):
    """Fit the model and evaluate it using various metrics with a timeout."""
    start_time = time.time()
    
    try:
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        if elapsed_time > timeout:
            print(f"Model {model.__class__.__name__} fitting exceeded timeout of {timeout} seconds.")
            return
        
        y_pred = model.predict(X_test)
        
        # Check for probability predictions
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        print(f"Model: {model.__class__.__name__}")
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        if y_pred_proba is not None:
            if len(np.unique(y_test)) > 2:  # Handle multi-class case
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
                print("ROC AUC Score (multi-class): ", roc_auc)
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                print("ROC AUC Score: ", roc_auc)
        
        print("F1 Score (weighted): ", f1_score(y_test, y_pred, average='weighted'))
        print("-" * 60)

    except Exception as e:
        print(f"An error occurred while evaluating model {model.__class__.__name__}: {str(e)}")

# Preprocessing pipelines for numeric and categorical data
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessor to X_train and X_test
X_train_imputed = preprocessor.fit_transform(X_train)
X_test_imputed = preprocessor.transform(X_test)

# Convert the result back to DataFrame for interpretability (optional)
X_train_imputed_df = pd.DataFrame(X_train_imputed.toarray() if hasattr(X_train_imputed, 'toarray') else X_train_imputed)
X_test_imputed_df = pd.DataFrame(X_test_imputed.toarray() if hasattr(X_test_imputed, 'toarray') else X_test_imputed)

# Evaluate each model on the preprocessed data with a timeout for KNN specifically.
for name, model in models.items():
    if name == 'KNN':  # Apply timeout specifically for KNN
        evaluate_model_with_timeout(model, X_train_imputed_df, X_test_imputed_df, y_train, y_test)
    else:
        evaluate_model_with_timeout(model, X_train_imputed_df, X_test_imputed_df, y_train, y_test)

# Explainable AI with SHAP for RandomForest model (or any other model of choice)
explainer = shap.TreeExplainer(models['RandomForest'])
shap_values = explainer.shap_values(X_test_imputed_df)

# Plot SHAP summary plot for RandomForest (or any other model of choice)
shap.summary_plot(shap_values[1], X_test_imputed_df)

# Predict on rows with missing loan_status and save predictions to Excel
missing_loan_status = loan_data[loan_data['loan_status'].isnull()]
predictions = models['RandomForest'].predict(missing_loan_status.drop(columns=['loan_status']))
missing_loan_status['loan_status'] = predictions

missing_loan_status.to_excel('LoanStatusPredictions.xlsx', index=False)