import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data/Lead Scoring.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Remove columns with >40% missing values
# -----------------------------
missing = df.isnull().mean() * 100

drop_cols = missing[missing > 40].index

df.drop(columns=drop_cols, inplace=True)

# -----------------------------
# Fill Missing Values
# -----------------------------
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# Remove ID Columns
# -----------------------------
df.drop(columns=["Prospect ID", "Lead Number"], inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

# -----------------------------
# Features and Target
# -----------------------------
# -----------------------------
# Features and Target
# -----------------------------
X = df[[
    "Total Time Spent on Website",
    "TotalVisits",
    "Page Views Per Visit"
]]

y = df["Converted"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/lead_model.pkl")

print("\nModel Saved Successfully!")