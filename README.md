# 🚗 Vehicle Price Prediction App

An interactive Streamlit web application that predicts car prices based on various vehicle specifications and features using a linear regression model.

---

## 📌 Project Overview

This project explores and analyzes vehicle data to:

- 🔍 Understand which features most influence car pricing  
- 🧠 Build and deploy a machine learning model to predict MSRP (Manufacturer's Suggested Retail Price)  
- 📊 Provide visual insights using Plotly and Seaborn  
- 🌐 Create an easy-to-use interface using Streamlit  

---

## 📊 Features

- 🎛️ **Interactive UI**: Choose vehicle specs from the sidebar to get instant price predictions  
- 🔮 **Real-time Prediction**: Uses a trained `LinearRegression` model to predict MSRP  
- 📈 **Feature Importance**: Visual insights showing which features most affect price  
- 🧹 **Data Cleaning & Preprocessing**: Handles missing values, converts categorical variables, and extracts numeric values  
- 📉 **Model Evaluation**: Evaluates the model using RMSE, MAE, and R² metrics  

---

## 🧠 Machine Learning

- **Model**: `LinearRegression` from scikit-learn  
- **Feature Engineering**:
  - Extracted numeric horsepower and torque from string fields
  - One-hot encoded categorical variables
- **Feature Importance**: Calculated using `DecisionTreeClassifier`  

---

## 📁 Files

| File                       | Description                                     |
|---------------------------|-------------------------------------------------|
| `app.py`                  | Streamlit app source code                       |
| `car_data.csv`            | Raw dataset                                     |
| `feature_importance.xlsx` | Ranked feature importance from Decision Tree    |
| `linear_model.pkl`        | Trained regression model                        |
| `pic1.png`, `pic2.png`    | UI images for sidebar and banner                |
| `data_with_pred.xlsx`     | Dataset with predicted MSRP values              |

---

## 🚀 How to Run

1. **Clone the repository:**

git clone https://github.com/your-username/vehicle-price-prediction.git
cd vehicle-price-prediction

2. **Install dependencies:**
pip install -r requirements.txt

3. **Run the app:**
streamlit run app.py

🛠️ Tech Stack
Language: Python

Data Handling: Pandas, NumPy

Modeling: scikit-learn

Visualization: Plotly, Seaborn, Matplotlib

Web App: Streamlit

Image Handling: Pillow (PIL)
