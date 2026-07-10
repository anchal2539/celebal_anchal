import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Lead Scoring Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/lead_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'models/lead_model.pkl' exists.")
    st.stop()

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio("Go to", ["Single Lead Prediction", "Bulk Prediction & Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("""
### Machine Learning Project
**Developed Using:**
* Python
* Pandas
* Plotly
* Scikit-Learn
* Streamlit
""")

# ---------------------------------
# App Mode 1: Single Lead Prediction
# ---------------------------------
if app_mode == "Single Lead Prediction":
    st.title("📊 Lead Scoring Prediction System")
    st.markdown("Predict whether a single customer lead is likely to convert using your ML model.")
    st.divider()

    st.subheader("📝 Enter Lead Details")
    col1, col2 = st.columns(2)

    with col1:
        total_time = st.number_input("Total Time Spent on Website (seconds)", min_value=0, value=500)
    with col2:
        total_visits = st.number_input("Total Visits", min_value=0, value=5)

    page_views = st.number_input("Page Views Per Visit", min_value=0.0, value=2.0)
    st.divider()

    if st.button("🔍 Predict Lead", use_container_width=True):
        input_data = pd.DataFrame({
            "Total Time Spent on Website": [total_time],
            "TotalVisits": [total_visits],
            "Page Views Per Visit": [page_views]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📈 Prediction Result")
        
        if prediction[0] == 1:
            st.success("✅ Lead is likely to Convert")
        else:
            st.error("❌ Lead is NOT likely to Convert")

        st.metric(label="Conversion Probability", value=f"{probability * 100:.2f}%")
        st.progress(float(probability))

        if probability >= 0.80:
            st.info("🔥 High probability of conversion. Sales team should contact immediately.")
        elif probability >= 0.50:
            st.warning("⚠ Medium probability of conversion. Follow-up is recommended.")
        else:
            st.error("❌ Low probability of conversion. This lead has lower conversion chances.")

# ---------------------------------
# App Mode 2: Bulk Prediction & Dashboard
# ---------------------------------
elif app_mode == "Bulk Prediction & Dashboard":
    st.title("📊 Bulk Lead Analysis & Insights Dashboard")
    st.markdown("Upload a CSV file containing multiple leads to predict conversion scores and visualize data instantly.")
    st.divider()

    uploaded_file = st.file_uploader("📥 Upload Lead CSV File", type=["csv"])

    if uploaded_file is not None:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # Required columns check
        required_cols = ["Total Time Spent on Website", "TotalVisits", "Page Views Per Visit"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV file must contain these columns: {required_cols}")
        else:
            # 1. Extract features & Handle Missing Values (NaN) smoothly
            features = df[required_cols].copy()
            features = features.fillna(0)
            
            # 2. Predict using cleaned features
            df['Prediction'] = model.predict(features)
            
            # 3. Calculate and Safe-Map Probabilities for Plots
            probabilities = model.predict_proba(features)[:, 1]
            df['Conversion Probability (%)'] = (probabilities * 100).round(2)
            
            # 4. Map prediction to text status
            df['Status'] = df['Prediction'].map({1: 'Likely to Convert', 0: 'Not Likely'})

            # ---- Metrics Display ----
            total_leads = len(df)
            converted_leads = int(df['Prediction'].sum())
            conversion_rate = (converted_leads / total_leads) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Leads Processed", total_leads)
            m2.metric("Predicted Conversions", converted_leads)
            m3.metric("Overall Conversion Rate", f"{conversion_rate:.2f}%")
            
            st.divider()

            # ---- Charts Section ----
            st.subheader("📈 Visual Insights")
            c1, c2 = st.columns(2)

            with c1:
                # Conversion Split Pie Chart
                fig_pie = px.pie(df, names='Status', title='Lead Conversion Distribution', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                # Time Spent vs Probability Scatter Chart
                fig_scatter = px.scatter(df, x='Total Time Spent on Website', y='Conversion Probability (%)',
                                         color='Status', title='Time Spent vs. Conversion Probability',
                                         labels={'Total Time Spent on Website': 'Time on Site (seconds)'})
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()

            # ---- Data Table & Download ----
            st.subheader("📋 Detailed Prediction Results")
            st.dataframe(df, use_container_width=True)

            # Convert processed dataframe to CSV for download
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv_data,
                file_name="lead_predictions_output.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("ℹ Please upload a CSV file with lead data to generate the interactive dashboard.")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
