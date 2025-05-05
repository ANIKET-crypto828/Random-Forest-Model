# 🏦 Bank Customer Churn Prediction

This project builds a machine learning pipeline to classify bank customers as **likely** or **unlikely to leave** (churn), using transaction data. It leverages **Random Forest** for the baseline model and **XGBoost** to boost performance.

---

## 🚀 Overview

Customer churn is a major challenge in the banking sector. Predicting churn helps banks take proactive measures to retain valuable clients. This project uses a classification approach to:

- Analyze transaction behavior
- Predict churn likelihood
- Improve accuracy with XGBoost

---

## 📊 Dataset

- **Source**: Simulated bank transaction data (or [replace with actual dataset source if public])
- **Features**:
  - Account balance
  - Number of transactions
  - Average transaction value
  - Customer tenure
  - Credit score
  - ...and more

- **Target**:
  - `Exited` (1 = customer left, 0 = customer stayed)

---

## 🔧 Tools & Technologies

- Python 3.x
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib / seaborn (for visualization)
- Jupyter Notebook

---

## 🧠 Models Used

### 1. Random Forest Classifier
- Baseline model
- Handles nonlinear relationships
- Outputs initial feature importance

### 2. XGBoost Classifier
- Gradient boosting model
- Improves precision, recall, and F1-score
- Better handling of class imbalance
- More robust regularization

---

## 📈 Evaluation Metrics

- Accuracy
- Precision / Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 🗂️ Project Structure

📁 churn-prediction/
├── data/
│ └── bank_customers.csv
├── notebooks/
│ └── churn_modeling.ipynb
├── models/
│ └── rf_model.pkl
│ └── xgb_model.pkl
├── README.md
└── requirements.txt


---

## ✅ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/ANIKET-crypto828/Random-Forest-Model
   cd churn-prediction
2. Install Dependencies:
pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook notebooks/churn_modeling.ipynb

📌 Next Steps
Real-time scoring system integration

Model explainability using SHAP

Deploy via Flask API / Streamlit dashboard

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

📬 Contact
Author: [Aniket Santra]

LinkedIn: [https://www.linkedin.com/in/aniket-santra-980030275/]

Email: [aniketsantra78@gmail.com]


---

Would you like me to generate the `requirements.txt` file or a sample Jupyter notebook for this?
