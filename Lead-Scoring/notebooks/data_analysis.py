import pandas as pd

# Load dataset
df = pd.read_csv("data/Lead Scoring.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nShape of Dataset:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Information
print("\nDataset Info:")
df.info()

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical Summary
print("\nSummary:")
print(df.describe())

# Target Variable Count
print("\nConverted Count:")
print(df["Converted"].value_counts())