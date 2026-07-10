# 📊 Lead Scoring Prediction Dashboard

An interactive Machine Learning web application built using **Streamlit** and **Scikit-Learn** to predict and analyze customer lead conversion probabilities.

## 🚀 Features
* **Single Lead Prediction:** Enter individual lead parameters to instantly calculate conversion probability with dynamic risk/priority recommendations.
* **Bulk Lead Analytics Dashboard:** Upload a raw `Lead Scoring.csv` file to process thousands of leads simultaneously.
* **Interactive Visualizations:** Real-time data distribution insights using Plotly charts (Pie Chart for conversion split & Scatter Plot for engagement metrics).
* **Data Export:** Download the final predicted dataset as a fresh CSV with attached conversion scores and priority statuses.

## 🛠️ Project Structure
```text
Lead-Scoring/
│
├── models/
│   └── lead_model.pkl          # Trained Logistic Regression Model
│
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # System dependencies and packages
└── README.md                   # Project Documentation