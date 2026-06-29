import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("used_cars.csv")

print("\nFirst 5 Rows\n")
print(data.head())

print("\nDataset Information\n")
print(data.info())

print("\nMissing Values\n")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())

# Remove Duplicate Rows
data.drop_duplicates(inplace=True)

# Fill Missing Values
for col in ["fuel_type", "accident", "clean_title"]:
    data[col] = data[col].fillna(data[col].mode()[0])

# Clean Price Column
data["price"] = (
    data["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Clean Mileage Column
data["milage"] = (
    data["milage"]
    .str.replace(",", "", regex=False)
    .str.replace(" mi.", "", regex=False)
    .astype(float)
)

# Encode Categorical Columns
encoder = LabelEncoder()

categorical_columns = [
    "brand",
    "model",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
    "clean_title"
]

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Graph 1 : Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(data["price"], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Graph 2 : Fuel Type Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=data["fuel_type"])
plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.show()

# Graph 3 : Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Graph 4 : Mileage vs Price
plt.figure(figsize=(7,5))
plt.scatter(data["milage"], data["price"])
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs Price")
plt.show()

# Features and Target
X = data.drop("price", axis=1)
y = data["price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance\n")

print("MAE :", round(mean_absolute_error(y_test, y_pred),2))
print("MSE :", round(mean_squared_error(y_test, y_pred),2))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)),2))
print("R2 Score :", round(r2_score(y_test, y_pred),4))

# Graph 5 : Actual vs Predicted Price
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance\n")
print(importance)

# Graph 6 : Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Predict Sample Car
sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPredicted Price for Sample Car : $", round(prediction[0],2))