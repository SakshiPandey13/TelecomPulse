# TelecomPulse: IBM Telco Customer Churn Prediction

## 1. Project Overview

TelecomPulse is a data analysis and machine learning project that explores customer churn in the telecommunications industry. It uses historical customer data to identify churn patterns, investigate factors associated with customer departures, and compare classification algorithms for predicting whether a customer is likely to leave a service provider.

The project combines data preprocessing, exploratory data analysis (EDA), machine learning, model evaluation, and business recommendations to demonstrate how customer data can support customer-retention planning.

## 2. Problem Statement

Customer churn can affect recurring revenue, customer relationships, and long-term business growth. Telecommunications companies need to understand which customer groups are more likely to leave and identify patterns that can inform retention strategies.

TelecomPulse addresses this problem by analyzing customer characteristics, examining their relationships with churn, and evaluating machine learning models that predict customer churn.

## 3. Project Objectives

The project has five main objectives:

1. Analyze customer churn patterns and understand the distribution of customers who stay or leave.
2. Investigate the relationship between contract type, customer tenure, and churn.
3. Examine customer charges, subscribed services, and payment methods in relation to churn.
4. Implement and compare four classification algorithms using multiple evaluation metrics.
5. Identify important predictive features and recommend practical customer-retention strategies.

## 4. Dataset

- **Dataset:** IBM Telco Customer Churn
- **Source:** [Kaggle — Telco Customer Churn]https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
- **Total records:** 7,043 customers
- **Original columns:** 21
- **Target variable:** `Churn`
- **Target classes:** `Yes` and `No`

The dataset includes customer account information, contract types, tenure, subscribed services, payment methods, and charges.

The original dataset is stored in `data/raw/`, while the cleaned dataset is stored in `data/processed/`.

## 5. Data Cleaning and Preprocessing

The following preprocessing steps were completed:

- Inspected dataset dimensions, column names, data types, missing values, and duplicate records.
- Converted `TotalCharges` from text to numeric values.
- Identified 11 blank `TotalCharges` values associated with customers whose tenure was zero and whose churn status was `No`; these values were set to zero.
- Verified that the cleaned dataset contained no missing values or duplicate records.
- Preserved the original raw dataset.
- Removed `customerID` from the model inputs because it is an identifier.
- Encoded the target variable and categorical input features.
- Split the dataset into training and testing sets using a 70:30 ratio, `random_state=42`, and stratification.
- Applied feature scaling for Logistic Regression and KNN.

## 6. Exploratory Data Analysis

Multiple visualizations were created to investigate customer behavior and churn patterns.

### Key Findings

**Overall churn distribution**

- Customers who stayed: 5,174 (approximately 73.5%).
- Customers who churned: 1,869 (approximately 26.5%).

**Contract type**

Observed churn rates were approximately 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts.

**Customer tenure**

Churn was concentrated among customers with shorter tenure, highlighting the importance of understanding the early customer experience.

**Monthly charges**

Customers who churned tended to have higher monthly charges than customers who stayed.

**Internet service**

Fiber-optic customers accounted for the highest observed number of churned customers among the internet-service categories examined.

**Payment method**

Electronic-check customers had the highest observed churn rate, at approximately 45%, among the payment methods examined.

**Numerical relationships**

The correlation analysis showed a strong positive relationship between tenure and `TotalCharges` (approximately 0.83) and a positive relationship between `MonthlyCharges` and `TotalCharges` (approximately 0.65).

These findings describe patterns in the dataset and do not establish that individual characteristics directly cause churn.

## 7. Machine Learning Models

Four classification algorithms were implemented and compared:

1. **Logistic Regression** — estimates the probability of a customer belonging to the churn class.
2. **Decision Tree** — uses feature-based decision rules to classify customers.
3. **Random Forest** — combines multiple decision trees and provides feature-importance estimates.
4. **K-Nearest Neighbors (KNN)** — classifies customers based on nearby observations in the feature space.

All four models were evaluated using the same 70:30 train-test split.

## 8. Model Evaluation and Results

The models were evaluated using accuracy, churn precision, churn recall, churn F1-score, macro F1-score, balanced accuracy, and confusion matrices.

### Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 73.26% | 49.65% | 50.98% | 50.31% |
| Random Forest | 78.70% | 62.82% | 48.48% | 54.73% |
| Logistic Regression | **80.41%** | **65.03%** | **56.68%** | **60.57%** |
| KNN | 74.35% | 51.74% | 50.27% | 50.99% |

### Results Interpretation

Logistic Regression achieved the highest accuracy, churn precision, churn recall, and churn F1-score among the four models in the current experiment.

Its accuracy was 80.41%, while its churn recall was 56.68%. These results make Logistic Regression the leading model in this comparison.

Random Forest provides an additional perspective through its feature-importance analysis, which helps identify variables useful for predicting churn.

Model performance is evaluated on the held-out test data. Accuracy alone is not sufficient for churn prediction because identifying customers who may leave is also important.

## 9. Feature Importance

Random Forest feature importance was used to examine which input variables contributed most to the model's predictions.

The leading features included:

- `TotalCharges`
- `MonthlyCharges`
- `tenure`
- `Contract`

Other features, including payment method and online security, also contributed to the model.

These rankings describe predictive usefulness within the fitted model. They do not prove that a feature causes customer churn.

The feature-importance visualization is available in `visualisation/11_feature_importance.png`.

## 10. Business Recommendations

Based on the observed patterns, the project suggests several potential customer-retention actions.

- **Improve early customer engagement:** Provide onboarding assistance and follow-ups during the initial months of a subscription.
- **Explore contract-retention offers:** Investigate suitable long-term plans, loyalty benefits, and flexible offers for month-to-month customers.
- **Review pricing and perceived value:** Examine whether customers with higher monthly charges are satisfied with the services they receive.
- **Investigate payment experiences:** Explore why electronic-check customers have a higher observed churn rate.
- **Prioritize retention outreach:** Use validated churn predictions to help customer-support teams decide which customers may need proactive attention.

These recommendations are opportunities for further investigation and testing, rather than guaranteed solutions.

## 11. Project Structure

```text
Telco_Customer_Churn_Prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_customer_churn_cleaned.csv
│
├── notebooks/
│   └── data_inspection.py
│
├── visualisation/
│   ├── 01_customer_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_churn_by_tenure.png
│   ├── 04_monthly_charges_by_churn.png
│   ├── 05_churn_by_internet_service.png
│   ├── 06_churn_by_payment_method.png
│   ├── 07_churn_rate_by_payment_method.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_decision_tree.png
│   ├── 10_tenure_vs_total_charges.png
│   └── 11_feature_importance.png
│
├── result/
│   └── model_comparison.csv
│
├── reports/
│   └── project_report.md
│
├── screenshots/
├── README.md
├── requirements.txt
└── .gitignore
```

## 12. Tools and Technologies

- **Programming language:** Python
- **Data manipulation:** pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Development environment:** Spyder and Visual Studio Code
- **Version control:** Git and GitHub

## 13. How to Run the Project

1. Clone or download the repository.
2. Open the project folder in Visual Studio Code or Spyder.
3. Install the required dependencies listed in `requirements.txt`.
4. Ensure the raw dataset is available at the expected path under `data/raw/`.
5. Run `notebooks/data_inspection.py` using the configured Python environment.
6. Review the generated visualizations, model comparison results, and project report.

The script uses file paths defined in the project code, so run it from the project root or ensure the paths are configured correctly.

## 14. Detailed Report

The detailed report is available at [`reports/project_report.md`](reports/project_report.md).

It describes the problem statement, project objectives, preprocessing, exploratory analysis, model evaluation, key findings, business recommendations, and future scope.

## 15. Project Status

**Current status:** Data cleaning, exploratory data analysis, four-model comparison, feature-importance analysis, and project documentation have been completed.

The current model comparison uses a 70:30 train-test split with `random_state=42` and stratification.

## 16. Disclaimer

This project is an educational data-science project using the IBM Telco Customer Churn dataset. Results describe patterns in the available data and model performance under the stated experimental setup. Predictions are estimates, not guarantees of customer behavior, and the recommendations require further evaluation before real-world implementation.
