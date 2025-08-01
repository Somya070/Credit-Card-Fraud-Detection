import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("credit_card_fraud_xgb_model.pkl")

st.title("Credit Card Fraud Detection App")

st.write("Upload a CSV file with the same columns as the training dataset (except Class).")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Agar 'Class' column accidentally upload ho gaya ho toh remove kar do
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # Prediction
    predictions = model.predict(data)

    data['Prediction'] = predictions

    st.write("Predictions (0 = Legit, 1 = Fraud):")
    st.write(data)

    fraud_count = (predictions == 1).sum()
    st.success(f"Total Fraudulent Transactions Detected: {fraud_count}")

    # ================== GRAPHS ==================

    st.subheader("Graphs")

    # Fraud vs Legit counts
    counts = pd.Series(predictions).value_counts()

    # 1. Bar chart (styled)
    st.write("### Fraud vs Legit Transactions")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", ax=ax1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Legit (0)', 'Fraud (1)'])
    ax1.set_ylabel("Count")
    ax1.set_title("Fraud vs Legit Transactions", fontsize=14, fontweight='bold')
    sns.despine()
    st.pyplot(fig1)

    # 2. Pie chart (styled)
    st.write("### Fraud vs Legit Pie Chart")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        counts.values,
        labels=['Legit', 'Fraud'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#6EC6FF', '#FF6B6B'],
        wedgeprops={'edgecolor': 'white'}
    )
    ax2.set_title("Fraud vs Legit Ratio", fontsize=14, fontweight='bold')
    st.pyplot(fig2)
