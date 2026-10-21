
import pandas as pd

# Load the Telco Customer Churn dataset
file_path = r"D:\Telco_Customer_Churn_Prediction\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"

data = pd.read_csv(file_path)
print(data.describe())
print(data.info())
# Display the first five records
print(data.head())

# Display dataset dimensions
print("Dataset shape:", data.shape)

# Check missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Total missing values
print("\nTotal missing values:")
print(data.isnull().sum().sum())

# Check duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Check data types
print("\nData types:")
print(data.dtypes)

# Check unique values in the target column
print("\nChurn distribution:")
print(data["Churn"].value_counts())

# Check for blank or whitespace-only TotalCharges entries
print("\nBlank TotalCharges entries:")
print(data["TotalCharges"].astype(str).str.strip().eq("").sum())

# Step 3: Clean TotalCharges

# Convert blank strings to missing values, then convert to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"].astype(str).str.strip().replace("", pd.NA),
    errors="coerce"
)

# Check the data type after conversion
print("\nTotalCharges data type:")
print(data["TotalCharges"].dtype)

# Count missing values after conversion
print("\nMissing TotalCharges values:")
print(data["TotalCharges"].isnull().sum())

# Display records where TotalCharges is missing
print("\nRecords with missing TotalCharges:")
print(data[data["TotalCharges"].isnull()])

# Check tenure for these records
print("\nTenure of records with missing TotalCharges:")
print(data.loc[data["TotalCharges"].isnull(), "tenure"])

# Step 4: Handle missing TotalCharges values

# Replace missing TotalCharges values with 0 for customers with zero tenure as they the affected customers have zero number of tenure
data.loc[
    data["TotalCharges"].isnull() & (data["tenure"] == 0),
    "TotalCharges"
] = 0

# Check missing values after handling them
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Check the total number of missing values
print("\nTotal missing values after cleaning:")
print(data.isnull().sum().sum())

# Verify the records that were previously missing
print("\nPreviously missing TotalCharges records after cleaning:")
print(data.loc[[488, 753, 936], ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])


# Step 5: Checking unique values in categorical columns
# This helps us check columns such as gender, Contract, PaymentMethod, and InternetService for inconsistent labels, unexpected categories, or extra whitespace.

# Select all categorical columns
categorical_columns = data.select_dtypes(include="object").columns

# Display unique values in each categorical column
for column in categorical_columns:
    print(f"\nUnique values in {column}:")
    print(data[column].unique())

# Step 6: Final checks and save the cleaned dataset

# Check the final dataset dimensions
print("\nFinal dataset shape:")
print(data.shape)

# Check the total number of missing values
print("\nTotal missing values:")
print(data.isnull().sum().sum())

# Check the number of duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Save the cleaned dataset
output_path = r"D:\Telco_Customer_Churn_Prediction\data\processed\telco_customer_churn_cleaned.csv"

data.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved location:", output_path)
