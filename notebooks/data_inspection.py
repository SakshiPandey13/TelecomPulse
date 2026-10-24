import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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


# Visualization 1: Customer Churn Distribution
# Define the folder for saving visualizations
visualization_path = r"D:\Telco_Customer_Churn_Prediction\visualisation"
os.makedirs(visualization_path, exist_ok=True)

# Define colors
churn_colors = {
    "No": "#16B8A6",
    "Yes": "#FF6B6B"
}

# Calculate customer counts
churn_counts = data["Churn"].value_counts().reindex(["No", "Yes"])

# Display counts and percentages
print("Customer Churn Counts:")
print(churn_counts)

print("\nCustomer Churn Percentages:")
print((churn_counts / churn_counts.sum() * 100).round(2))

# Create the visualization
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#F8F9FF")

bars = ax.bar(
    ["Stayed", "Churned"],
    churn_counts.values,
    color=[churn_colors["No"], churn_colors["Yes"]],
    width=0.55
)

# Display values above the bars
ax.bar_label(bars, fmt="%d", padding=6, fontsize=11, fontweight="bold")

ax.set_title("Customer Churn Distribution", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Customer Status", fontsize=11)
ax.set_ylabel("Number of Customers", fontsize=11)

# Add percentage labels inside the bars
for bar, count in zip(bars, churn_counts.values):
    percentage = count / churn_counts.sum() * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 0.80,
        f"{percentage:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

sns.despine()
plt.tight_layout()

# Save the visualization
plt.savefig(
    os.path.join(visualization_path, "01_customer_churn_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#Observation:
#The chart shows that 73.5% of customers stayed with the company,while 26.5% churned. This indicates that approximately one-fourth
#of the customers left, highlighting the need for effective customer retention strategies.


# Visualization 2: Churn by Contract Type
# Calculate total customers in each contract type
total_customers = data.groupby("Contract").size()

# Calculate customers who churned in each contract type
churned_customers = data[data["Churn"] == "Yes"].groupby("Contract").size()

# Calculate churn percentage
contract_churn = (churned_customers / total_customers) * 100

print("Churn Rate by Contract Type:")
print(contract_churn.round(2))

# Create a bar chart
plt.figure(figsize=(8, 5))

plt.bar(
    contract_churn.index,
    contract_churn.values,
    color=["#16B8A6", "#FF6B6B", "#6C5CE7"]
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

# Save the chart
plt.savefig(
    os.path.join(visualization_path, "02_churn_by_contract.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Observation:
# Customers with month-to-month contracts have the highest churn rate,followed by one-year contracts. Two-year contracts have the lowest
# churn rate, suggesting that longer contracts are associated with better customer retention.

# Visualization 3: Customer Tenure vs Churn
# Separate customers who stayed and churned
stayed = data[data["Churn"] == "No"]
churned = data[data["Churn"] == "Yes"]

# Create a histogram
plt.figure(figsize=(8, 5))

plt.hist(
    stayed["tenure"],
    bins=24,
    color="#16B8A6",
    alpha=0.7,
    label="Stayed"
)

plt.hist(
    churned["tenure"],
    bins=24,
    color="#FF6B6B",
    alpha=0.7,
    label="Churned"
)

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()

plt.tight_layout()

# Save the chart
plt.savefig(
    os.path.join(visualization_path, "03_churn_by_tenure.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Observation:
# Customer churn is concentrated in the early months of tenure.
# Customers who remain with the company tend to have longer tenure,highlighting the importance of retaining new customers.

# Visualization 4: Monthly Charges by Churn
# Separate monthly charges by churn status
stayed_charges = data[data["Churn"] == "No"]["MonthlyCharges"]
churned_charges = data[data["Churn"] == "Yes"]["MonthlyCharges"]

# Create a box plot
plt.figure(figsize=(8, 5))

plt.boxplot(
    [stayed_charges, churned_charges],
    tick_labels=["Stayed", "Churned"],
    patch_artist=True,
    boxprops=dict(facecolor="#16B8A6"),
    medianprops=dict(color="black", linewidth=2)
)

plt.title("Monthly Charges by Customer Churn")
plt.xlabel("Customer Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()

# Save the chart
plt.savefig(
    os.path.join(visualization_path, "04_monthly_charges_by_churn.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Observation:
# Customers who churned generally had higher monthly charges than
# customers who stayed. Pricing may be one factor associated with
# customer churn.

# Visualization 5: Churn by Internet Service
# Count customers who stayed and churned for each internet service
stayed = data[data["Churn"] == "No"]
churned = data[data["Churn"] == "Yes"]

stayed_counts = stayed["InternetService"].value_counts()
churned_counts = churned["InternetService"].value_counts()

# Combine the counts
internet_counts = pd.DataFrame({
    "Stayed": stayed_counts,
    "Churned": churned_counts
}).fillna(0)

# Create a bar chart
internet_counts.plot(
    kind="bar",
    color=["#16B8A6", "#FF6B6B"],
    figsize=(8, 5)
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Customer Status")
plt.tight_layout()

# Save the chart
plt.savefig(
    os.path.join(visualization_path, "05_churn_by_internet_service.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Observation:
# Fiber-optic customers account for the highest number of churned
# customers, followed by DSL customers. This suggests that the company
# should investigate customer satisfaction and service quality among
# fiber-optic users.


# Visualization 6: Churn by Payment Method

# Separate customers who stayed and churned
stayed = data[data["Churn"] == "No"]
churned = data[data["Churn"] == "Yes"]

# Count customers for each payment method
stayed_counts = stayed["PaymentMethod"].value_counts()
churned_counts = churned["PaymentMethod"].value_counts()

# Combine the counts
payment_counts = pd.DataFrame({
    "Stayed": stayed_counts,
    "Churned": churned_counts
}).fillna(0)

# Create a bar chart
payment_counts.plot(
    kind="bar",
    color=["#16B8A6", "#FF6B6B"],
    figsize=(10, 5)
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Customer Status")
plt.tight_layout()

# Save the chart
plt.savefig(
    os.path.join(visualization_path, "06_churn_by_payment_method.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Observation:
# Electronic check users have the highest number of churned customers.
# The company should investigate whether payment convenience or customer experience contributes to this pattern.
# Step 1: Prepare Data for Machine Learning

# Remove customerID because it is only an identifier
model_data = data.drop(columns=["customerID"])

# Convert Churn into numerical values
model_data["Churn"] = model_data["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate input features and target variable
X = model_data.drop(columns=["Churn"])
y = model_data["Churn"]

# Display the dataset information
print("\nInput Features Shape:", X.shape)
print("Target Variable Shape:", y.shape)

print("\nTarget Variable Distribution:")
print(y.value_counts())

print("\nFirst five rows of input features:")
print(X.head())

# Step 2: Split Data into Training and Testing Sets

from sklearn.model_selection import train_test_split

# Split the dataset while preserving the churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the sizes of the datasets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())
