# 🩺 Diabetes Prediction App

A web application that uses Machine Learning to predict the likelihood of diabetes based on patient health metrics. Built with **Streamlit** and **Scikit-learn**.

---

## ✨ Features

- **Machine Learning Model**: Gradient Boosting Classifier trained on healthcare dataset
- **Interactive UI**: Simple and responsive interface built with Streamlit
- **Real-time Prediction**: Instant results with risk probability and analysis
- **History Tracking**: Save and export prediction history
- **Visual Analytics**: Gauge charts and risk factor analysis

---

## 📊 Dataset

### Features Description
| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| `gender` | Patient's gender | Categorical | Male, Female |
| `age` | Age in years | Numeric | 0.08 - 80 |
| `hypertension` | High blood pressure | Binary | 0 (No), 1 (Yes) |
| `heart_disease` | Heart disease | Binary | 0 (No), 1 (Yes) |
| `smoking_history` | Smoking history | Categorical | never, former, current, etc. |
| `bmi` | Body Mass Index | Numeric | 10.0 - 95.7 |
| `HbA1c_level` | Hemoglobin A1c level | Numeric | 3.5 - 9.0 |
| `blood_glucose_level` | Blood glucose level | Numeric | 80 - 300 |
| `diabetes` | Diagnosis (target) | Binary | 0 (No), 1 (Yes) |

### Target Distribution
```python
# 0: 50.1% (Non-Diabetic)
# 1: 49.9% (Diabetic)
```
---
### 🤖 Model Performance
```bash
Model	Accuracy	ROC-AUC
Gradient Boosting	92%	0.982
Random Forest	91%	0.976
XGBoost	91%	0.978
Logistic Regression	89%	0.966
```
---
### 📁 Project Structure
```bash
text
Diabetes_Prediction/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages
├── setup.sh                  # Streamlit configuration
├── .gitignore                # Git ignore file
│
├── gradient_boosting_model.pkl   # Trained ML model
├── scaler.pkl                     # Feature scaler
├── gender_encoder.pkl             # Gender encoder
├── smoking_encoder.pkl            # Smoking encoder
├── imputer.pkl                     # Categorical imputer
└── median_imputer.pkl              # Numerical imputer
```
---
### ⚙️ Technologies Used
```bash
Python 3.9+

Streamlit - Web framework

Scikit-learn - Machine learning

Pandas & NumPy - Data processing

Plotly - Interactive visualizations

Pickle/Joblib - Model serialization
```
---
### 🚀 How to Run
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/saifalaswad43/diabetes-prediction-app.git
cd diabetes-prediction-app
```
### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
### 4️⃣ Open in Browser
```text
http://localhost:8501
```
---
### 🌐 Live Demo
- https://diabetes-predictionml-ml.streamlit.app/
---
### 🧠 Machine Learning Workflow
- Data Cleaning - Handle missing values and outliers
- Data Preprocessing - Encode categorical variables
- Feature Scaling - Standardize numerical features
- Model Training - Train multiple classifiers
- Model Evaluation - Compare performance metrics
- Model Saving - Export best model using pickle
- Web Deployment - Deploy with Streamlit
---
### 📈 App Features
- Home Page: Model statistics and performance metrics
- Prediction Page: Patient data input form with instant results
- History Page: Save and export prediction history
- About Page: Project information and developer details
---
### 👨‍💻 Developer
``` bash
Saif Alaswad
📧 Email: saifalaswad43@gmail.com
🐙 GitHub: @saifalaswad43
```
