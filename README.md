# 🩺 Diabetes Prediction App

A simple web application that uses Machine Learning to predict the likelihood of diabetes based on patient health metrics.

---
### Dataset Features
| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| `gender` | Patient's gender | Categorical | Male, Female, Other |
| `age` | Age in years | Numeric | 0.08 - 80 years |
| `hypertension` | High blood pressure | Binary | 0 (No), 1 (Yes) |
| `heart_disease` | Heart disease | Binary | 0 (No), 1 (Yes) |
| `smoking_history` | Smoking history | Categorical | never, No Info, former, current, not current, ever |
| `bmi` | Body Mass Index | Numeric | 10.0 - 95.7 |
| `HbA1c_level` | Hemoglobin A1c level | Numeric | 3.5 - 9.0 |
| `blood_glucose_level` | Blood glucose level | Numeric | 80 - 300 |
| `diabetes` | Diabetes diagnosis (target) | Binary | 0 (No), 1 (Yes) |
### Exploratory Data Analysis
```python
# Target distribution
diabetes_distribution = df['diabetes'].value_counts()
# 0: 50.1% (Non-Diabetic)
# 1: 49.9% (Diabetic)
```
## ✨ Features

- **Machine Learning Model**: Accurate prediction model trained on healthcare datasets
- **Web Interface**: Simple and responsive UI built using HTML and CSS  
- **Real-time Prediction**: User inputs are processed instantly through a Flask API

---

## 📁 Project Structure
Diabetes_Prediction/
│
├── app.py # Flask backend server
├── notebook.ipynb # Data cleaning and model training
├── model.pkl # Trained ML model
├── scaler.pkl # Data scaler
├── encoder.pkl # Encoder if used
│
└── templates/
└── index.html # Frontend interface

text

---

## ⚙️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- HTML
- CSS
- Joblib

---
## 🚀 How to Run

### 1️⃣ Install Requirements
```bash
Open your terminal and run:
### pip install flask flask-cors pandas scikit-learn joblib
```
### 2️⃣ Run the Application
```bash
run streamlit app.py
```
### 3️⃣ Open in Browser
```bash
Go to:
text
http://127.0.0.1:5000
```
---
### 🧠 Machine Learning Workflow
- Data Cleaning
- Data Preprocessing
- Feature Scaling
- Model Training
- Model Saving using joblib
- Deploying the model using Flask
---
### 👨‍💻 Developer
Saif Alaswad

