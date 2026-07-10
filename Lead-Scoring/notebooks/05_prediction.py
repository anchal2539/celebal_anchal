import joblib
import pandas as pd

# Load Model
model = joblib.load("models/lead_model.pkl")

# Load Dataset
df = pd.read_csv("data/Lead Scoring.csv")

# Remove columns with >40% missing values
missing = df.isnull().mean() * 100
drop_cols = missing[missing > 40].index
df.drop(columns=drop_cols, inplace=True)

# Fill Missing Values
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove ID Columns
df.drop(columns=["Prospect ID", "Lead Number"], inplace=True)

# Encode Categorical Columns
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = encoder.fit_transform(df[col])

# Features
X = df.drop("Converted", axis=1)

# Prediction
prediction = model.predict(X.head(10))

print("Predictions:")
print(prediction)