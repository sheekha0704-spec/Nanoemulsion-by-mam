import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Nano-emulsion ML Tool", layout="wide")

st.title("🧪 Nano-emulsion Formulation Predictor")
st.markdown("Predict **Droplet Size**, **PDI**, and **Stability** based on formulation parameters.")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Using the filename from your second notebook
    df = pd.read_csv('drug_loaded_nanoemulsion_1500 (1).csv')
    return df

try:
    df = load_data()
    st.sidebar.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Please upload 'drug_loaded_nanoemulsion_1500 (1).csv' to your GitHub repo. Error: {e}")
    st.stop()

# --- 2. Sidebar Inputs ---
st.sidebar.header("Formulation Parameters")

# Categorical inputs based on your dataset columns
oil_type = st.sidebar.selectbox("Oil Type", df['Oil_Type'].unique())
surf_type = st.sidebar.selectbox("Surfactant Type", df['Surfactant_Type'].unique())
drug_name = st.sidebar.selectbox("Drug Name", df['Drug_Name'].unique())

# Numerical inputs (min/max based on data ranges)
sys_hlb = st.sidebar.slider("System HLB", float(df['System_HLB'].min()), float(df['System_HLB'].max()), 10.0)
drug_load = st.sidebar.number_input("Drug Loading (mg/mL)", value=15.0)
ee_percent = st.sidebar.slider("EE Percent", 0.0, 100.0, 80.0)
smix = st.sidebar.number_input("Smix Ratio", value=1.0)

# --- 3. Model Training ---
# Defining features as per your notebook logic
X = df.drop(columns=['ID', 'PDI', 'Stability_Target'])
y_pdi = df['PDI']
y_stability = df['Stability_Target']

categorical_cols = ['Oil_Type', 'Surfactant_Type', 'Drug_Name']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# PDI Pipeline
pdi_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Stability Pipeline
stb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

@st.cache_resource
def train_models(_X, _y_pdi, _y_stb):
    pdi_model.fit(_X, _y_pdi)
    stb_model.fit(_X, _y_stb)
    return pdi_model, stb_model

pdi_mdl, stb_mdl = train_models(X, y_pdi, y_stability)

# --- 4. Prediction Logic ---
if st.button("Generate Prediction"):
    # Creating input dataframe to match training features
    # Note: We fill missing columns with average/placeholder if not in sidebar
    input_dict = {col: df[col].mean() if df[col].dtype != 'object' else df[col].mode()[0] for col in X.columns}
    
    # Update with user inputs
    input_dict.update({
        'Oil_Type': oil_type,
        'Surfactant_Type': surf_type,
        'Drug_Name': drug_name,
        'System_HLB': sys_hlb,
        'Drug_Loading_mg_mL': drug_load,
        'EE_Percent': ee_percent,
        'Smix_Ratio': smix
    })
    
    input_df = pd.DataFrame([input_dict])
    
    pdi_pred = pdi_mdl.predict(input_df)[0]
    stb_pred = stb_mdl.predict(input_df)[0]
    stability_label = "✅ Stable" if stb_pred == 1 else "❌ Unstable"
    
    col1, col2 = st.columns(2)
    col1.metric("Predicted PDI", f"{pdi_pred:.4f}")
    col2.metric("Stability Status", stability_label)

# --- 5. Data Insights ---
if st.checkbox("Show Feature Importance (PDI)"):
    # Accessing the regressor inside the pipeline
    importances = pdi_mdl.named_steps['regressor'].feature_importances_
    # Note: Feature names change after OneHotEncoding, for simplicity we plot raw values
    st.bar_chart(importances)
