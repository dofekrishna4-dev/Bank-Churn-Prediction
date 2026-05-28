import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar
st.sidebar.title("🏦 Bank Churn ML App")

st.sidebar.info(
    """
    This project predicts customer churn using Machine Learning.

    Built with:
    - Streamlit
    - Scikit-learn
    - Pandas
    """
)
# Page Config
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load model
model = joblib.load("models/churn_model.pkl")

# Title
st.title("🏦 Bank Churn Prediction System")
st.markdown("Predict whether a customer will leave the bank or not.")

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("📋 Customer Information")

    credit_score = st.slider("Credit Score", 300, 900, 600)

    age = st.slider("Age", 18, 100, 40)

    tenure = st.slider("Tenure", 0, 10, 5)

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        max_value=250000.0,
        value=50000.0
    )

    products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

with col2:

    st.subheader("📌 Additional Details")

    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=200000.0,
        value=50000.0
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

st.divider()

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "81%")
col2.metric("Model", "Logistic Regression")
col3.metric("Dataset Size", "10,000")

# Prediction Button
if st.button("🔍 Predict Churn"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary],
        'Geography_Germany': [1 if geography == "Germany" else 0],
        'Geography_Spain': [1 if geography == "Spain" else 0],
        'Gender_Male': [1 if gender == "Male" else 0]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    # Result
    if prediction == 1:
        st.error(f"❌ Customer is likely to churn")

    else:
        st.success(f"✅ Customer is likely to stay")

    st.subheader("📊 Churn Probability")

    st.progress(float(probability))

    st.write(f"Probability of churn: **{probability:.2%}**")

    # Dashboard Chart
    st.subheader("📈 Customer Overview")

    chart_data = pd.DataFrame({
        "Feature": [
            "CreditScore",
            "Age",
            "Balance",
            "Salary"
        ],
        "Value": [
            credit_score,
            age,
            balance,
            salary
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(chart_data["Feature"], chart_data["Value"])

    plt.xticks(rotation=0)

    st.pyplot(fig)

    st.divider()

st.markdown(
    """
    ### 👨‍💻 Developed by Krishna Dofe

    End-to-End Machine Learning Project using:
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    """
)