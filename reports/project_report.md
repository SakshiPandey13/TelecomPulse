# TelecomPulse: Customer Churn Prediction and Analysis

## 1. Introduction

Customer churn is an important challenge in the telecommunications industry. When customers discontinue their subscriptions, companies may lose recurring revenue and need to spend additional resources acquiring new customers. Understanding customer behavior can help businesses identify patterns associated with churn and develop better customer-retention strategies.

TelecomPulse is a data analysis and machine learning project that investigates customer churn using the IBM Telco Customer Churn dataset. The project combines data preprocessing, exploratory data analysis, visualization, and classification algorithms to understand customer behavior and predict whether a customer is likely to leave a telecom service.

The project aims to transform customer data into meaningful insights that can support informed business decisions and proactive customer-retention planning.

## 2. Problem Statement

Telecommunication companies serve customers with different contracts, service subscriptions, payment methods, tenure periods, and spending patterns. Identifying the characteristics associated with customer churn can help companies understand which customer groups may require additional attention.

The problem addressed by TelecomPulse is to analyze customer churn patterns, investigate the relationships between customer characteristics and churn, compare machine learning classification algorithms, and identify useful predictive features that can support customer-retention decisions.

## 3. Project Objectives

The project has five main objectives.

### Objective 1: Analyze customer churn patterns

To examine the distribution of customers who stayed with the company and those who discontinued their services.

This objective helps quantify customer churn and understand the overall scale of customer loss.

### Objective 2: Investigate contract type, tenure, and churn

To analyze how customer churn varies according to contract type and customer tenure.

This helps identify customer groups, such as month-to-month customers and customers in their early subscription period, that may deserve closer attention.

### Objective 3: Examine charges, subscribed services, and payment methods

To investigate the relationships between churn and customer characteristics, including monthly charges, total charges, internet services, and payment methods.

This objective helps reveal patterns that can guide further investigation into customer experience, billing, and service satisfaction.

### Objective 4: Implement and compare four classification algorithms

To build and evaluate four machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

The models are compared using accuracy, churn precision, churn recall, churn F1-score, macro F1-score, and balanced accuracy.

### Objective 5: Identify important predictive factors and recommend retention strategies

To examine feature importance, identify variables that contribute to prediction, and translate the findings into practical recommendations that a telecom company could investigate or test.

## 4. Dataset Description

The project uses the IBM Telco Customer Churn dataset.

- **Number of records:** 7,043 customers
- **Number of original columns:** 21
- **Target variable:** `Churn`
- **Target categories:** `Yes` and `No`

The dataset contains information about customer demographics, account details, contract types, subscribed services, payment methods, tenure, and charges.

The `Churn` column indicates whether a customer left the company. It is the target variable used by the classification models.

## 5. Data Cleaning and Preprocessing

Data preparation was performed before exploratory analysis and model development.

The following steps were completed:

1. Loaded the original customer dataset and examined its structure.
2. Inspected data types, missing values, and duplicate records.
3. Converted `TotalCharges` from text to a numeric data type.
4. Identified 11 blank `TotalCharges` values. These records had zero tenure and a churn status of `No`, so their total charges were set to zero.
5. Checked the cleaned dataset for missing values and duplicate records.
6. Preserved the original raw dataset and saved the cleaned data separately.
7. Removed `customerID` from the model inputs because it is an identifier rather than a customer-behavior feature.
8. Converted the target variable into numerical values, with `No = 0` and `Yes = 1`.
9. Encoded categorical input features using `LabelEncoder`.
10. Divided the data into training and testing sets using a 70:30 split, with `random_state=42` and stratification to preserve the target-class proportions.

The cleaned dataset contains 7,043 records and 21 columns, with no missing values or duplicate records after preprocessing.

## 6. Exploratory Data Analysis

Exploratory data analysis was conducted using charts and summary statistics to investigate customer churn and its relationships with different customer characteristics.

### 6.1 Overall customer churn distribution

Out of 7,043 customers:

- Customers who stayed: 5,174, approximately 73.5%.
- Customers who churned: 1,869, approximately 26.5%.

The analysis shows that approximately one-quarter of the customers in the dataset discontinued their services. This provides a baseline for understanding the scale of churn in the dataset.

### 6.2 Contract type and churn

The observed churn rates differ substantially across contract types:

- Month-to-month contracts: approximately 42.7%.
- One-year contracts: approximately 11.3%.
- Two-year contracts: approximately 2.8%.

Month-to-month customers have the highest observed churn rate. Customers with longer-term contracts have lower observed churn rates.

These findings suggest that contract type is an important area for customer-retention analysis. Companies could investigate whether flexible contract options, loyalty benefits, and appropriate long-term offers help improve retention.

### 6.3 Customer tenure and churn

The tenure analysis shows that churn is concentrated among customers with shorter subscription periods. Customers who have stayed with the company for longer periods are generally more likely to remain.

This finding highlights the importance of the early customer experience. Onboarding assistance, service guidance, and timely follow-ups could be considered during the initial months of a subscription.

### 6.4 Monthly charges and churn

The analysis of monthly charges shows that customers who churn tend to have higher monthly charges than customers who stay.

This pattern suggests that pricing and perceived value may be useful areas for further investigation. However, the relationship does not establish that higher charges directly cause churn.

### 6.5 Internet services and churn

The analysis compares customer churn across internet-service categories. Fiber-optic customers account for the highest observed number of churned customers in the dataset.

Customer counts and churn rates represent different measures. The number of churned customers indicates the size of a group, while the churn rate shows the proportion of that group that left.

These findings can help a company investigate service satisfaction, pricing, and customer expectations across internet-service categories.

### 6.6 Payment method and churn

Electronic-check customers have the highest observed churn rate, at approximately 45%, among the payment methods examined.

This pattern provides a reason to investigate the payment experience, customer preferences, and possible friction during billing. It does not establish that the payment method itself causes customers to leave.

### 6.7 Relationships between numerical features

The correlation heatmap reveals several relationships between numerical variables:

- Tenure and TotalCharges: approximately 0.83.
- MonthlyCharges and TotalCharges: approximately 0.65.
- Tenure and MonthlyCharges: approximately 0.25.

The strong positive relationship between tenure and TotalCharges is consistent with the fact that customers who remain subscribed for longer periods can accumulate higher total charges.

These relationships help explain the structure of the customer data and provide context for interpreting the predictive models.

## 7. Machine Learning Methodology

Four classification algorithms were implemented to predict whether a customer would churn.

### 7.1 Logistic Regression

Logistic Regression estimates the probability that a customer belongs to a particular class. In this project, it is used to classify customers as churned or not churned.

### 7.2 Decision Tree

The Decision Tree model uses a series of feature-based decisions to classify customers. Its tree structure can help illustrate how customer characteristics are used to produce predictions.

### 7.3 Random Forest

Random Forest combines predictions from multiple decision trees. It is also used to estimate feature importance, helping identify which variables are most useful to the model.

### 7.4 K-Nearest Neighbors (KNN)

KNN classifies a customer based on the classes of nearby observations in the feature space. Standardization was applied to the model inputs used by KNN so that differences in numerical scales would have less influence on distance calculations.

### 7.5 Training and testing procedure

The dataset was divided into 70% training data and 30% testing data. All four models used the same split so that their results could be compared under the same experimental conditions.

The training data was used to fit the models, while the testing data was used to evaluate predictions on records not used for model fitting.

Standard scaling was applied to the inputs for Logistic Regression and KNN. Decision Tree and Random Forest were trained using the encoded features.

## 8. Model Evaluation and Comparison

The models were evaluated using six metrics.

- **Accuracy:** The percentage of all predictions that were correct.
- **Churn precision:** The proportion of customers predicted to churn who actually churned.
- **Churn recall:** The proportion of actual churners correctly identified by the model.
- **Churn F1-score:** The harmonic mean of churn precision and churn recall.
- **Macro F1-score:** The average F1-score across both target classes, giving each class equal weight.
- **Balanced accuracy:** The average recall across the two target classes.

### 8.1 Results

The following results were obtained using the 70:30 train-test split.

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 73.26% | 49.65% | 50.98% | 50.31% |
| Random Forest | 78.70% | 62.82% | 48.48% | 54.73% |
| Logistic Regression | 80.41% | 65.03% | 56.68% | 60.57% |
| KNN | 74.35% | 51.74% | 50.27% | 50.99% |

### 8.2 Interpretation of the results

Logistic Regression achieved the highest accuracy, churn precision, churn recall, and churn F1-score among the four models in this experiment.

Its accuracy of 80.41% indicates that approximately 80 out of every 100 test-set predictions were correct. Its churn recall of 56.68% indicates that it identified approximately 57 out of every 100 actual churners in the test set.

Random Forest achieved a churn precision of 62.82% and an accuracy of 78.70%. Its feature-importance output also provides a way to examine which customer characteristics are useful for prediction.

Decision Tree and KNN provide additional comparison points and demonstrate how different classification approaches perform on the same dataset.

Based on the current results, Logistic Regression is the leading model for this experiment. Model selection should consider the business objective as well as accuracy, especially when identifying customers who may leave is important.

## 9. Key Findings: What TelecomPulse Tells Us

The project provides five main insights for people interested in customer churn and retention.

### Finding 1: It quantifies customer churn

The dataset contains 1,869 churned customers out of 7,043 records, corresponding to a churn rate of approximately 26.5%.

This gives the company a clear baseline for understanding the extent of customer loss in the observed data.

### Finding 2: It identifies customer groups associated with higher churn

Month-to-month customers and customers with shorter tenure show higher observed churn. These patterns help identify customer groups that may benefit from closer attention and further analysis.

### Finding 3: It reveals patterns in customer spending and services

Monthly charges, internet-service categories, and payment methods show different relationships with churn. These findings can guide investigations into pricing, service quality, billing experiences, and customer satisfaction.

### Finding 4: It compares approaches for predicting customer churn

The project evaluates four classification algorithms under the same 70:30 split. Logistic Regression achieves the strongest overall results among the four models based on the reported accuracy, churn precision, churn recall, and churn F1-score.

This provides a measurable basis for selecting a model for further evaluation.

### Finding 5: It identifies predictive features and supports retention planning

The Random Forest feature-importance analysis ranks `TotalCharges`, `MonthlyCharges`, and `tenure` among the most important predictive features in the current model.

Combined with the exploratory findings, this helps translate customer data into practical questions and potential retention strategies.

Feature importance indicates predictive usefulness; it does not prove that a feature causes churn.

## 10. Business Recommendations

The findings suggest several actions that telecommunications companies could consider testing.

### 10.1 Strengthen early customer engagement

Because churn is concentrated among customers with shorter tenure, companies could provide better onboarding, service guidance, and early follow-ups to understand customer concerns.

### 10.2 Develop appropriate contract-retention offers

The higher observed churn rate among month-to-month customers suggests an opportunity to test suitable longer-term contract options, loyalty benefits, or retention offers.

Such offers should be evaluated for customer satisfaction and financial value rather than assuming that every customer wants a longer contract.

### 10.3 Investigate pricing and perceived value

Since churned customers tend to have higher monthly charges, companies could examine whether customers believe they are receiving sufficient value for the price paid. Clear billing information and appropriate service options may improve the customer experience.

### 10.4 Review payment and service experiences

The high observed churn rate among electronic-check customers and differences across internet-service categories provide useful starting points for further investigation.

Companies could examine payment-related issues, service satisfaction, and customer support experiences before deciding which interventions to introduce.

### 10.5 Use predictions to prioritize retention outreach

A validated churn model could help customer-support teams prioritize outreach to customers who may be at risk of leaving.

Predictions should be combined with customer feedback and business rules. The effectiveness of retention actions should be measured to determine whether they actually improve customer retention.

## 11. Conclusion

TelecomPulse demonstrates how data analysis and machine learning can be used to understand customer churn and support customer-retention planning in the telecommunications industry.

The project analyzed 7,043 customer records, cleaned and prepared the data, explored churn patterns, and investigated relationships involving contract type, tenure, charges, internet services, and payment methods.

The analysis found that month-to-month customers and customers with shorter tenure have higher observed churn rates. It also identified useful patterns involving monthly charges and payment methods. These findings provide a basis for investigating customer needs and designing targeted retention strategies.

Four classification algorithms—Logistic Regression, Decision Tree, Random Forest, and KNN—were implemented and evaluated. Logistic Regression achieved the highest accuracy of 80.41%, churn precision of 65.03%, churn recall of 56.68%, and churn F1-score of 60.57% in the current 70:30 experiment.

The Random Forest feature-importance analysis highlighted TotalCharges, MonthlyCharges, and tenure as important predictive features. Together, the exploratory analysis and predictive modeling provide complementary views of customer behavior: the analysis explains observed patterns, while the models estimate which customers may churn.

The main contribution of TelecomPulse is its end-to-end approach to transforming customer data into interpretable findings, model comparisons, and practical business recommendations. These outputs can help a telecom company investigate customer-retention opportunities and make more informed decisions.

## 12. Limitations and Future Scope

Future improvements could include:

- Evaluating alternative categorical encoding methods to examine their effect on model performance.
- Testing different classification thresholds to explore the trade-off between churn precision and churn recall.
- Using additional evaluation methods to assess how consistently the models perform.
- Examining customer feedback and service-quality data to gain a deeper understanding of churn behavior.
- Testing retention strategies with real business data to measure their actual effect on customer retention.

These extensions could help develop a more robust predictive system and improve the practical usefulness of the project.

## 13. Project Deliverables

The TelecomPulse project includes:

- Raw and cleaned customer datasets.
- Data inspection and preprocessing code.
- Exploratory data analysis visualizations.
- Customer churn analysis by contract, tenure, charges, services, and payment method.
- Four trained classification models.
- A model comparison using multiple evaluation metrics.
- Random Forest feature-importance analysis.
- Visual outputs supporting the analysis.
- Business recommendations based on the observed findings.

Together, these deliverables demonstrate a complete student data-science workflow, from data preparation and exploration to predictive modeling and business interpretation.
