# Loan Default Prediction Analysis

## Overview

In this project, we aim to analyze loan data to predict whether borrowers will default on their loans. By leveraging various machine learning models, we explore the relationships within the data, preprocess it effectively, and evaluate multiple models to identify the best-performing one. Our analysis focuses on understanding the factors that contribute to loan defaults in order to assist lenders in making informed decisions.

## Table of Contents

- [Project Goals](#project-goals)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing and Feature Engineering](#data-processing-and-feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Results and Insights](#results-and-insights)
- [Contributing](#contributing)
- [License](#license)

## Project Goals

We aim to achieve the following objectives:

1. **Data Preprocessing**: Clean and prepare the data for analysis.
2. **Feature Engineering**: Create new features that can improve model performance.
3. **Model Training and Evaluation**: Train multiple machine learning models and evaluate their performance.
4. **Hyperparameter Tuning**: Optimize model parameters for better accuracy.
5. **Insights Generation**: Provide insights into which features are most important in predicting loan defaults.

## Data Description

We utilize two datasets for this analysis:

1. **MPLCaseStudy.csv**: Contains information about loans, including borrower details, loan amounts, interest rates, and loan statuses.
2. **mapping.csv**: Provides mappings for column names to ensure clarity in our analysis.

## Installation

To run this project, we need to set up our environment with the necessary libraries. We recommend using a virtual environment. Here’s how you can set it up:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>


2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install required packages:
   ```bash
   pip install -r requirements.txt

## Usage

To run the analysis, execute the following command in your terminal:
   ```bash
   python modelsv2_moremodels.py
   ```
This will execute the script, which performs data loading, preprocessing, model training, evaluation, and generates insights.


## Data Processing and Feature Engineering

In this section of our code, we focus on cleaning the data and creating meaningful features:

- We handle missing values by employing techniques such as median imputation for numeric columns.
- We convert date columns into a datetime format for easier manipulation.
- Percentage strings are transformed into float values for numerical analysis.
- New features are engineered based on existing data (e.g., calculating years since a borrower's earliest credit line).

## Model Evaluation

We explore several machine learning models to determine which performs best in predicting loan defaults:

1. Random Forest Classifier
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Gradient Boosting Classifier
6. Decision Tree Classifier
7. AdaBoost Classifier
8. Bagging Classifier
9. XGBoost Classifier
10. LightGBM Classifier
11. CatBoost Classifier

Each model is evaluated based on metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.

## Results and Insights

After evaluating our models, we found that:

- The Random Forest Classifier achieved an accuracy of approximately 97%, demonstrating its robustness in predicting loan defaults.
- The Logistic Regression model performed well with an accuracy of about 94%, but it struggled with identifying defaults effectively compared to Random Forest.
- The KNN model had the lowest accuracy of approximately 86%, indicating significant weaknesses in its predictive capabilities.

### Key Takeaways:

- Random Forest is recommended as our primary model due to its high accuracy and robustness.
- Logistic Regression can be used for interpretability alongside Random Forest.
- KNN may not be suitable for this dataset given its performance issues.
