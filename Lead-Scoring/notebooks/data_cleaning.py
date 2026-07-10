import pandas as pd

# Load Dataset
df = pd.read_csv("data/Lead Scoring.csv")

print("Original Shape:", df.shape)

# Calculate missing percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Percentage:")
print(missing_percentage.sort_values(ascending=False))

# Remove columns having more than 40% missing values
df = df.loc[:, missing_percentage < 40]

print("\nShape After Removing Columns:", df.shape)
print("\nRemaining Columns:")
print(df.columns)

# Fill numerical missing values with median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check missing values again
print("\nRemaining Missing Values:")
print(df.isnull().sum())