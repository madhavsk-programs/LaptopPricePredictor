# 💻 Laptop Price Predictor

This project predicts the price of a laptop based on its specifications using Machine Learning.  
It uses an end-to-end ML pipeline with **feature engineering, ensemble learning (Random Forest, XGBoost, Gradient Boosting, ExtraTrees)** and is deployed with **Streamlit** for an interactive UI.  

---

## 🚀 Features
- Predict laptop prices from key specs:
  - Brand
  - Type (Ultrabook, Gaming, Notebook, etc.)
  - RAM & Weight
  - Touchscreen & IPS Display
  - Screen Size & Resolution (PPI calculated automatically)
  - CPU & GPU
  - HDD & SSD Storage
  - Operating System
- End-to-end ML pipeline with preprocessing + model
- Handles log-transformed targets → converts back to rupees for prediction
- Interactive **Streamlit web app** for easy use

---

## 🛠 Tech Stack
- **Python 3.11**
- **Pandas, NumPy, Scikit-learn**
- **XGBoost, Random Forest, Gradient Boosting, ExtraTrees**
- **Streamlit** (for deployment)

---
