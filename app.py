import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Insight Pro", layout="centered")

# --- STYLE ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://fluentsupport.com/wp-content/uploads/2023/07/Build-customer-loyalty.jpg");
        background-size: cover;
        color: white;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    h1 { text-align: center; color: #00d4ff; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ---
with open('churn_model.pkl', 'rb') as f:
    assets = pickle.load(f)
model, le_contract, le_internet = assets["model"], assets["le_contract"], assets["le_internet"]

st.markdown("<h1> Customer Loyalty Intelligence</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.number_input("Tenure (Months)", 1, 100, 12)
    with col2:
        charges = st.number_input("Monthly Charges ($)", 0, 3000, 500)
        contract = st.selectbox("Contract", le_contract.classes_)
    
    internet = st.selectbox("Internet Service", le_internet.classes_)
    
    if st.button("RUN ANALYSIS"):
        c_code = le_contract.transform([contract])[0]
        i_code = le_internet.transform([internet])[0]
        features = [[age, tenure, charges, c_code, i_code]]
        
        prob = model.predict_proba(features)[0][1] * 100
        
        # --- GAUGE CHART ---
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            title = {'text': "Churn Probability (%)", 'font': {'size': 24, 'color': "white"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00d4ff"},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 40], 'color': '#22c55e'},
                    {'range': [40, 70], 'color': '#eab308'},
                    {'range': [70, 100], 'color': '#ef4444'}],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig)
        
        if prob > 50:
            st.error(f"High Churn Risk: {prob:.1f}%")
        else:
            st.success(f"Customer is Likely to Stay: {100-prob:.1f}% Loyalty")
            
    st.markdown('</div>', unsafe_allow_html=True)