# churn
import pandas as pd
import numpy as np

# Setting the random seed for reproducibility
np.random.seed(42)

# Number of customers in the dataset
n_customers = 100000

# Generating customer data
customer_ids = range(1, n_customers + 1)
genders = np.random.choice(['Male', 'Female'], size=n_customers, p=[0.5, 0.5])
ages = np.random.randint(18, 70, size=n_customers)
tenure = np.random.randint(1, 73, size=n_customers)  # Tenure in months (1 to 72 months)
contracts = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], size=n_customers, p=[0.4, 0.3, 0.3])
monthly_charges = np.random.randint(20, 150, size=n_customers)  # Monthly charges between 20 and 150
churn = np.random.choice([0, 1], size=n_customers, p=[0.7, 0.3])  # 30% chance of churn

# Creating the DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Tenure': tenure,
    'Contract': contracts,
    'MonthlyCharges': monthly_charges,
    'Churn': churn
})

# Display the first few rows of the dataset
print(df.head())

# Saving the dataframe to a CSV file
df.to_csv('synthetic_customer_churn.csv', index=False)
import pandas as pd
import numpy as np

# Creating synthetic customer churn data
np.random.seed(42)

n_customers = 1000
customer_ids = range(1, n_customers + 1)
genders = np.random.choice(['Male', 'Female'], size=n_customers, p=[0.5, 0.5])
ages = np.random.randint(18, 70, size=n_customers)
tenure = np.random.randint(1, 73, size=n_customers)
contracts = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], size=n_customers, p=[0.4, 0.3, 0.3])
monthly_charges = np.random.randint(20, 150, size=n_customers)
churn = np.random.choice([0, 1], size=n_customers, p=[0.7, 0.3])

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Tenure': tenure,
    'Contract': contracts,
    'MonthlyCharges': monthly_charges,
    'Churn': churn
})

# Checking for missing values
df.isnull().sum()

# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)


from sklearn.model_selection import train_test_split

# Features and target variable
X = df.drop(columns=['Churn', 'CustomerID'])
y = df['Churn']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on test set
y_pred_logreg = logreg.predict(X_test)

# Evaluate model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

from sklearn.ensemble import RandomForestClassifier

# Initialize and train random forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Evaluate model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Feature importance from Random Forest
feature_importances = rf.feature_importances_
features = X.columns

# Create a DataFrame for easier visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

