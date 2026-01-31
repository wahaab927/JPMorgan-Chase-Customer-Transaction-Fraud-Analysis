# 📊 JPMorgan Chase Customer Transaction Fraud Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Portfolio%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This repository contains a **Customer Transaction Fraud Analysis** project by ABDUL WAHAAB modeled after a real-world banking risk scenario at **JPMorgan Chase**.

The project simulates how a Data Analyst or Risk Analyst would analyze transaction data, identify suspicious patterns, and build a machine learning model to classify transactions as **fraudulent** or **normal**.

The project demonstrates:
- Data cleaning and preparation  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Predictive modeling  
- Model evaluation  
- A reusable fraud prediction function  

This project is designed for **portfolio demonstration**, interview preparation, and hands-on practice with financial data analysis.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Flow & Architecture](#-data-flow--architecture)
- [API / Function Documentation](#-api--function-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## ❓ Problem Statement

Banks process thousands of transactions every day. While most are legitimate, some transactions may be fraudulent, causing financial losses and customer risk.

In this project:

- The dataset contains customer transaction details such as:
  - Transaction amount  
  - Transaction type  
  - Customer demographics  
  - Fraud label (`is_fraud` column)

- The objective is to:
  - Analyze patterns in fraudulent vs normal transactions  
  - Build a classification model to predict fraud  
  - Provide a function that predicts fraud for new transactions  

### 🎯 Objective
Create a system that:
- Identifies suspicious transaction behavior  
- Predicts fraud probability  
- Helps banks reduce financial risk  

---

## ✨ Features

- Fetches and processes customer transaction data  
- Performs Exploratory Data Analysis (EDA)  
- Visualizes fraud vs non-fraud patterns  
- Computes key metrics and features  
- Trains a machine learning classification model  
- Evaluates model using accuracy and classification metrics  
- Provides a reusable fraud prediction function  
- Modular and well-documented code structure  
- Portfolio-ready professional project  

---

## 🛠 Technology Stack

**Programming Language**
- Python 3.7+

**Libraries Used**
- pandas  
- numpy  
- matplotlib / seaborn  
- scikit-learn  

**Optional Tools**
- Jupyter Notebook  

---

## ⚙️ Installation

To set up this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/JPMorgan-Chase-Fraud-Analysis.git
cd JPMorgan-Chase-Fraud-Analysis
````

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

---

## ▶️ Usage

Follow these steps to use the project:

* Run the Python script:

```bash
python fraud_detection.py
```

* The script will:

  * Load and clean the dataset
  * Perform EDA
  * Train the fraud detection model
  * Display evaluation metrics
  * Enable fraud prediction on new transactions

* You may also use Jupyter Notebook for interactive analysis and visualization.

---

## 📁 Project Structure

A typical directory structure:

```
JPMorgan-Chase-Fraud-Analysis/
│
├── fraud_detection.py
├── transactions.csv
├── Problem_Statement_Fraud_Detection.txt
├── requirements.txt
├── README.md
└── tests/
    └── test_fraud_detection.py
```

### File Description

* `fraud_detection.py` → Main script containing data analysis, model training, and prediction function
* `transactions.csv` → Sample dataset of customer transactions
* `Problem_Statement_Fraud_Detection.txt` → Project description and requirements
* `requirements.txt` → Python dependencies
* `tests/` → Test cases for core functions

---

## 🔄 Data Flow & Architecture

The simulation follows a modular and readable data pipeline.
Main flow:

* **Data Ingestion:** Load transaction data from CSV
* **Data Processing:** Clean data and handle missing values
* **Analysis:** Perform EDA and identify fraud patterns
* **Modeling:** Train classification model
* **Evaluation:** Measure performance
* **Prediction:** Predict fraud for new transactions

### Simplified Data Flow

```
User Input
    ↓
Load Transaction Data
    ↓
Clean & Preprocess Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Feature Engineering
    ↓
Train ML Model
    ↓
Evaluate Model
    ↓
Predict Fraud
```

---

## 📘 API / Function Documentation

### Function: `predict_fraud(transaction_details)`

Predicts whether a transaction is fraudulent.

#### Description:

Takes transaction details as input and returns:

* Probability of fraud
* Predicted class (1 = Fraud, 0 = Normal)

#### Input Example:

```python
transaction_details = {
    "amount": 2500,
    "transaction_type": 1,
    "customer_age": 35,
    "account_balance": 15000
}
```

#### Output Example:

```json
{
  "fraud_probability": 0.78,
  "prediction": 1
}
```

Where:

* `1` = Fraud
* `0` = Normal

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

These metrics help assess the performance of the fraud detection system.

---

## 🧪 Testing

To ensure correctness of the code, run the test suite (if present):

```bash
pytest tests/
```

Test cases cover:

* Data loading
* Fraud prediction function
* Model output validation

---

## 🤝 Contributing

Contributions are welcome!

To contribute:

* Fork this repository
* Create a new branch
* Make your changes
* Commit with clear messages
* Submit a pull request

Please follow existing code style and add tests if required.

---

## 🛠 Troubleshooting

* Ensure all dependencies are installed
* Check Python version compatibility
* Verify dataset file path
* Restart kernel if using Jupyter Notebook
* Review error logs for debugging
* Ensure input format is correct for prediction function

---

## 📜 License

This repository is licensed under the **MIT License**.
See the `LICENSE` file for more details.

---

## 🙏 Acknowledgements

* Inspired by real-world banking fraud detection scenarios at JPMorgan Chase
* Built using open-source Python libraries such as pandas and scikit-learn
* Thanks to the data science community for learning resources and support

---

## 👨‍💼 Author

**Data Analyst Portfolio Project**

Developed by ABDUL WAHAAB to demonstrate skills in:

* Data Analysis
* Machine Learning
* Fraud Detection
* Financial Risk Analytics
**4) Make this README more visually attractive**
```
