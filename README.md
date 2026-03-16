# 🩺 Diabetes Prediction App

A simple web application that uses Machine Learning to predict the likelihood of diabetes based on patient health metrics.

---

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
Open your terminal and run:

```bash
pip install flask flask-cors pandas scikit-learn joblib
2️⃣ Run the Application
bash
python app.py
3️⃣ Open in Browser
Go to:

text
http://127.0.0.1:5000
📊 Dataset
The model was trained using a diabetes dataset containing several health indicators such as:

Glucose level

Blood pressure

BMI

Age

Insulin

Skin thickness

Pregnancies

🧠 Machine Learning Workflow
Data Cleaning

Data Preprocessing

Feature Scaling

Model Training

Model Saving using joblib

Deploying the model using Flask

👨‍💻 Developer
Saif Alaswad

