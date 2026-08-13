# 📈 Rossmann Sales Forecasting


Predict daily sales for Rossmann stores using Machine Learning and an interactive Streamlit dashboard. The application helps estimate sales based on store information, promotions, holidays, competition, and calendar features.


![Python](https://img.shields.io/badge/Python-3.11-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)

![License](https://img.shields.io/badge/License-MIT-blue)

##  Live Demo

[Launch the Application](https://project-6nhis-g7rvxumstx9916ofuyu4h5.streamlit.app/)


## 📷 Dashboard Preview

![Dashboard](images/dashboard.png)

## Project Overview

Rossmann operates thousands of drug stores across Europe. Accurate daily sales forecasting helps optimize inventory management, staffing, and promotional planning.

This project builds a Machine Learning pipeline that predicts daily store sales using historical transaction data, store information, promotional campaigns, holidays, and competition details.

The solution is deployed as an interactive Streamlit application where users can modify store attributes and instantly receive sales predictions with business insights.
---

## Business Problem

Rossmann operates thousands of retail stores. Accurate sales forecasting helps:

- Inventory Planning
- Staff Scheduling
- Promotion Planning
- Revenue Forecasting
- Better Business Decisions

---

## Dataset

The project uses the Rossmann Store Sales dataset containing:

- Store Information
- Promotion Details
- Competition Information
- Holidays
- Historical Sales
- Calendar Features

---

##  Features

- Interactive Streamlit Dashboard
- Real-time Sales Prediction
- Feature Importance Visualization
- Business Insight Cards
- Prediction Confidence Gauge
- Data Preprocessing Pipeline
- Random Forest Regression Model
- Optimized Deployment Model

---

## Machine Learning Pipeline

1. Data Cleaning
2. Missing Value Treatment
3. Feature Engineering
4. Label Encoding
5. Model Training
6. Model Evaluation
7. Streamlit Deployment

##  Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regression |
| MAE | 664.80 |
| RMSE | 1089.45 |
| R² Score | 0.9197 |

---

##  Streamlit Dashboard

The dashboard allows users to:

- Select Store Information
- Enter Competition Details
- Choose Prediction Date
- Predict Daily Sales
- View Processed Inputs
- Display Interactive Prediction Cards

---

### Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib
- Git
- GitHub

---

##  Installation

Clone the repository

```bash
git clone https://github.com/dubeyswati69/Project-6_NHIS.git
```

Install requirements

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

##  Folder Structure

```text
Project-6_NHIS
│
├── images/
│   └── dashboard.png
│
├── app.py
├── requirements.txt
├── sales_forecasting_streamlit.pkl
├── Rossmann_Sales_Forecasting.ipynb
├── README.md
└── .gitignore
```

##  Future Improvements

- Deploy on Streamlit Cloud
- Hyperparameter Optimization
- Model Monitoring
- Better Feature Engineering
- Interactive Business Dashboard

---

## Author

**Swati Dwivedi**

Data Analyst | Python | Machine Learning | Streamlit | SQL

GitHub: https://github.com/dubeyswati69

LinkedIn: https://www.linkedin.com/in/swati-dwivedi-667b50172/