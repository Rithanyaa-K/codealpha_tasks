import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Unemployment in India.csv")

data.columns = data.columns.str.strip()

print("\nFirst 5 Rows:\n")
print(data.head())

print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nSummary Statistics:\n")
print(data.describe())

sns.histplot(data['Estimated Unemployment Rate (%)'], bins=20)
plt.title("Unemployment Rate Distribution")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Count")
plt.show()

state_avg = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

state_avg.plot(kind='bar', figsize=(12, 6))
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Rate (%)")
plt.tight_layout()
plt.show()

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nAnalysis Completed Successfully!")