import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Customer CLV & Segmentation Engine", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('clv_segmentation_artifacts.pkl')

try:
    artifacts = load_artifacts()
    df = artifacts['dataset']
except Exception as e:
    st.error("Artifacts file 'clv_segmentation_artifacts.pkl' not found. Run the notebook/training script first.")
    st.stop()

st.title("📊 Customer Lifetime Value (CLV) & Segmentation Engine")
st.markdown("Combines **Unsupervised K-Means** and **Supervised Regression** to guide marketing spend and customer retention.")

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & CUSTOMER LOOKUP
# ---------------------------------------------------------
st.sidebar.header("🔍 Customer Search")
customer_ids = sorted(df['Customer_ID'].unique())
selected_id = st.sidebar.selectbox("Select or Type Customer ID:", customer_ids)

lookup_row = df[df['Customer_ID'] == selected_id].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Customer Details")
st.sidebar.write(f"**Customer ID:** {lookup_row['Customer_ID']}")
st.sidebar.write(f"**Segment:** {lookup_row['Segment_Name']}")
st.sidebar.write(f"**Predicted 90-Day Spend:** ${lookup_row['Predicted_90D_Spend']:,.2f}")
st.sidebar.info(f"**Action:** {lookup_row['Recommended_Action']}")

# ---------------------------------------------------------
# MAIN DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📌 Executive Summary", "🧩 Customer Segments", "🔮 PCA & Cluster Map"])

with tab1:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    total_pred_spend = df['Predicted_90D_Spend'].sum()
    avg_pred_spend = df['Predicted_90D_Spend'].mean()
    revenue_at_risk = df[df['Segment_Name'] == 'At-Risk / Lost']['Predicted_90D_Spend'].sum()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Proj. 90-Day Spend", f"${total_pred_spend:,.2f}")
    col3.metric("Avg Proj. Spend/Cust", f"${avg_pred_spend:,.2f}")
    col4.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}", delta="-At-Risk Segment")

    st.markdown("---")
    st.subheader("Revenue at Stake by Segment")
    segment_summary = df.groupby('Segment_Name').agg({
        'Customer_ID': 'count',
        'Monetary': 'mean',
        'Target_90D_Spend': 'sum',
        'Predicted_90D_Spend': 'sum'
    }).reset_index().rename(columns={
        'Customer_ID': 'Customer Count',
        'Monetary': 'Historical Avg Spend',
        'Target_90D_Spend': 'Actual 90D Spend',
        'Predicted_90D_Spend': 'Predicted 90D Spend'
    })
    
    st.dataframe(segment_summary.style.format({
        'Historical Avg Spend': '${:,.2f}',
        'Actual 90D Spend': '${:,.2f}',
        'Predicted 90D Spend': '${:,.2f}'
    }), use_container_width=True)

with tab2:
    st.subheader("Segment Breakdown & Recommended Actions")
    
    fig_seg = px.bar(
        segment_summary, 
        x='Segment_Name', 
        y='Predicted 90D Spend', 
        color='Segment_Name',
        text_auto='.2s',
        title="Predicted 90-Day Revenue Contribution by Segment"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

with tab3:
    st.subheader("2D PCA Projection of Customer Clusters")
    
    fig_pca = px.scatter(
        df, 
        x='PCA1', 
        y='PCA2', 
        color='Segment_Name',
        hover_data=['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Predicted_90D_Spend'],
        title="K-Means Clusters visualized via Principal Component Analysis (PCA)"
    )
    st.plotly_chart(fig_pca, use_container_width=True)