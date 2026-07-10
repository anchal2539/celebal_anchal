import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
df = pd.read_csv("data/Lead Scoring.csv")

# Create graphs folder if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# -------------------------------
# Graph 1: Lead Conversion Count
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Converted", data=df)

plt.title("Lead Conversion Count")
plt.xlabel("Converted")
plt.ylabel("Number of Leads")

plt.savefig("graphs/conversion_count.png")
plt.show()

# -------------------------------
# Graph 2: Top 10 Lead Sources
# -------------------------------

plt.figure(figsize=(10,6))

df["Lead Source"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Lead Sources")
plt.xlabel("Lead Source")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("graphs/lead_source.png")
plt.show()