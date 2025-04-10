# 📊 Customer Churn Prediction Model

![Project Banner](https://via.placeholder.com/1200x400.png?text=Customer+Churn+Prediction+System)

A machine learning model to predict customer churn using Telco customer data. Built with Python and Scikit-learn.

## Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Features
- Handles class imbalance using SMOTE
- Explains key churn factors using feature importance
- Achieves 78% prediction accuracy
- Includes comprehensive EDA visualizations
- Ready for deployment via API or dashboard

## 📁 Dataset
**Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Features:**
- 21 variables including:
  - Demographic info (gender, senior citizen status)
  - Account information (tenure, contract type)
  - Services subscribed (Internet, phone)
  - Billing details (monthly charges, payment method)

**Target Variable:** `Churn` (Yes/No)

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```
## Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

## Install dependencies:
```bash
pip install -r requirements.txt
```


## 📂 Project Structure
customer-churn-prediction/
├── data/                   # Dataset files
│   └── Customer_Churn.csv
├── notebooks/              # Jupyter notebooks
│   └── Customer_Churn_Analysis.ipynb
├── src/                    # Source code
│   ├── train_model.py
│   └── predict.py
├── models/                 # Saved models
├── visuals/                # Generated visualizations
├── requirements.txt        # Dependencies
└── README.md
