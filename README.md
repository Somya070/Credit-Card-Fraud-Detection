# Credit Card Fraud Detection App

This project is an **end-to-end Credit Card Fraud Detection system** built using **Machine Learning (XGBoost)** and deployed with an interactive **Streamlit Web App**.  
The app can detect fraudulent transactions and visualize predictions through graphs.

---

## 📊 Dataset
We used the publicly available **Credit Card Fraud Detection dataset**:
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

This dataset contains transactions made by European cardholders in September 2013, including:
- **Time, V1–V28 (PCA features), Amount, and Class (target variable)**  
- `Class = 1` → Fraud  
- `Class = 0` → Legitimate  

---

## 🚀 Features of the Project
1. **Data Preprocessing**
   - Handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.
   - Feature scaling and correlation analysis.

2. **Modeling**
   - Trained **XGBoost Classifier** on resampled data.
   - Evaluated model using precision, recall, F1-score.

3. **Streamlit Web Application**
   - Upload a CSV file (same format as training dataset except the `Class` column).
   - Get **real-time fraud predictions**.
   - **Visualizations**:
     - Fraud vs Legit transactions (Bar Chart + Pie Chart).

---

## 🛠 Tech Stack
- **Python**, Pandas, NumPy  
- **Machine Learning**: XGBoost, SMOTE (imbalanced-learn)  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Streamlit  

---

## 📂 Project Structure

credit-card-fraud-detection/
├── app.py # Streamlit app
├── credit_card_fraud_xgb_model.pkl # Trained model
├── requirements.txt
└── README.md


---

## 🔧 How to Run Locally

1. Clone the repository:
   
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   
2. Install dependencies:

   pip install -r requirements.txt

3. Run the Streamlit app:

   streamlit run app.py

## Live app demo:
https://credit-card-fraud-detection-dlh7bfzjdw6pbrxrkiwfu2.streamlit.app/
