import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Digital Health AI", layout="wide")

st.title("📱 Digital Health Risk Predictor")
st.markdown("""
### 🧠 What is this?

This AI-powered tool analyzes your daily digital habits and predicts potential health risks and productivity levels.

---

### ⚠️ Why does this matter?

Excessive screen time can lead to:
- Eye strain 👁️
- Sleep disturbances 😴
- Reduced concentration 🧠

This app helps you understand your risk and improve your lifestyle.

---
""")
st.markdown("### Analyze your screen habits, health & productivity")

df = pd.read_excel("synthetic_dataset.xlsx")
df = df.fillna("Unknown")

st.markdown("## 📊 Dataset Insights")
st.write("This section shows patterns observed in student screen usage and health effects.")
    
le_dict = {}
df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le

@st.cache_resource
def train_model():

    df_clean = df.copy()

    # Encode everything safely
    le_dict = {}

    for col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le

    X = df_clean.drop(['Issue', 'Productivity'], axis=1)
    y_issue = df_clean['Issue']
    y_prod = df_clean['Productivity']

    model_issue = RandomForestClassifier(n_estimators=200, max_depth=10)
    model_issue.fit(X, y_issue)

    model_prod = RandomForestClassifier(n_estimators=200, max_depth=10)
    model_prod.fit(X, y_prod)

    return model_issue, model_prod, X.columns, le_dict

model_issue, model_prod, features, le_dict = train_model()

st.markdown("## 🧠 Predict Your Health Risk")
st.write("Enter your lifestyle details below to get AI-based predictions.")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year", df['Year'].unique())
    age = st.slider("Age", 16, 30, 18)            
    branch = st.selectbox("Branch", df['Branch'].unique())
    gender = st.selectbox("Gender", df['Gender'].unique())
    screen = st.selectbox("Screen Time", df['ScreenTime'].unique())

with col2:
    purpose = st.selectbox("Purpose", df['Purpose'].unique())
    sleep = st.selectbox("Sleep", ["5-6 hours", "6-7 hours", "7+ hours"])  
    awareness = st.selectbox(
        "Awareness (knowledge about digital health)",
        ["Low", "Medium", "High"],
        help="How aware are you about screen-related health risks like eye strain, sleep issues, etc."
    )       
    tool = st.selectbox(
        "Tool Usage (protective tools)",
        ["Yes", "No"],
        help="Do you use tools like blue light filters, screen time trackers, or night mode?"
    )
    hygiene = st.selectbox("Digital Hygiene", ["Poor", "Average", "Good"])
    
if st.button("🚀 Predict"):

    input_dict = {
        'Year': year,
        'Age': age,
        'Branch': branch,
        'Gender': gender,
        'ScreenTime': screen,
        'Purpose': purpose,
        'Sleep': sleep,
        'Awareness': awareness,
        'ToolUsage': tool,
        'Hygiene': hygiene
    }

    input_df = pd.DataFrame([input_dict])

    for col in input_df.columns:
        if col in le_dict:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(
                lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else 0
            )
        
    issue_pred = model_issue.predict(input_df)[0]
    prod_pred = model_prod.predict(input_df)[0]

    issue = le_dict['Issue'].inverse_transform([issue_pred])[0]
    prod = le_dict['Productivity'].inverse_transform([prod_pred])[0]

    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⚠️ Health Issue", issue)

    with col2:
        st.metric("📈 Productivity", prod)

    risk_score = 0

    if "8" in screen:
        risk_score += 2
    elif "6" in screen:
        risk_score += 1

    if "5" in sleep:
        risk_score += 2
    elif "6" in sleep:
        risk_score += 1

    if awareness == "Low":
        risk_score += 2
    elif awareness == "Medium":
        risk_score += 1

    if risk_score >= 4:
        risk = "🔴 High"
    elif risk_score >= 2:
        risk = "🟡 Medium"
    else:
        risk = "🟢 Low"

    with col3:
        st.metric("🚨 Risk Level", risk)

    st.subheader("💡 Recommendations")

    if "8" in screen:
        st.error("Reduce screen time immediately!")

    if "5" in sleep:
        st.warning("Increase sleep to 7+ hours.")

    if awareness == "Low":
        st.warning("Improve awareness about digital health.")

    if hygiene == "Poor":
        st.warning("Practice better posture & breaks.")

    if tool == "No":
        st.info("Use blue light filters or screen trackers.")

    st.success("Stay healthy 🚀")
