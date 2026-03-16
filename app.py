import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Risk level badges */
    .risk-low {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    return smoking_status
@st.cache_resource
def load_models():
    """Load all saved models and preprocessors"""
    models = {}
    
    with st.spinner("🔄 Loading AI models..."):
        progress_bar = st.progress(0)
        files = {
            'imputer': 'imputer.pkl',
            'median_imputer': 'median_imputer.pkl',
            'gender_encoder': 'gender_encoder.pkl',
            'smoking_encoder': 'smoking_encoder.pkl',
            'scaler': 'scaler.pkl',
            'classifier': 'gradient_boosting_model.pkl'
        }
        
        for i, (key, file) in enumerate(files.items()):
            try:
                with open(file, 'rb') as f:
                    models[key] = pickle.load(f)
                progress_bar.progress((i + 1) * 16)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
                return None
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        
        st.success("✅ **AI Models Loaded Successfully!**")
    
    return models

def preprocess_input(data, models):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([data])
    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)
    missing_cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
    df[missing_cat_cols] = models['imputer'].transform(df[missing_cat_cols])
    median_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[median_cols] = models['median_imputer'].transform(df[median_cols])
    df['hypertension'] = df['hypertension'].astype(float)
    df['heart_disease'] = df['heart_disease'].astype(float)
    df['gender'] = models['gender_encoder'].transform(df['gender'])
    smoking_encoded = models['smoking_encoder'].transform(df[['smoking_history']]).toarray()
    smoking_cols = models['smoking_encoder'].get_feature_names_out(['smoking_history'])
    for i, col in enumerate(smoking_cols):
        df[col] = smoking_encoded[:, i]
    df = df.drop('smoking_history', axis=1)
    num_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 
                'HbA1c_level', 'blood_glucose_level']
    num_cols.extend(smoking_cols)
    final_df = pd.DataFrame()
    for col in num_cols:
        if col in df.columns:
            final_df[col] = df[col]
        else:
            final_df[col] = 0
    scaled_data = models['scaler'].transform(final_df)
    final_df_scaled = pd.DataFrame(scaled_data, columns=final_df.columns)
    
    return final_df_scaled

def predict_diabetes(models, input_data):
    """Make prediction using Gradient Boosting model"""
    try:
        processed_data = preprocess_input(input_data, models)
        prediction = models['classifier'].predict(processed_data)[0]
        probabilities = models['classifier'].predict_proba(processed_data)[0]
        probability = probabilities[1]  
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#28a745'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low", "🟢", "risk-low"
    elif probability < 0.6:
        return "Moderate", "🟡", "risk-moderate"
    else:
        return "High", "🔴", "risk-high"

def save_prediction(input_data, prediction, probability):
    """Save prediction to history"""
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'age': input_data['age'],
        'gender': input_data['gender'],
        'bmi': input_data['bmi'],
        'blood_glucose': input_data['blood_glucose_level'],
        'hba1c': input_data['HbA1c_level'],
        'hypertension': 'Yes' if input_data['hypertension'] == 1 else 'No',
        'heart_disease': 'Yes' if input_data['heart_disease'] == 1 else 'No',
        'smoking': input_data['smoking_history'],
        'prediction': "Diabetic" if prediction == 1 else "Non-Diabetic",
        'probability': f"{probability:.1%}"
    }
    st.session_state.prediction_history.append(record)

def main():
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🏥 Diabetes Risk Assessment Platform</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Powered by Advanced Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check the files.")
        st.stop()
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
        st.markdown("## **Navigation**")
        
        page = st.radio(
            "",
            ["🏠 Home", "🔮 Predict", "📊 History", "ℹ️ About"],
            index=0
        )
        st.markdown("---")
        if page == "🔮 Predict":
            st.markdown("### **Model Performance**")
            st.info("📈 **Accuracy:** 92%")
            st.info("🎯 **ROC-AUC:** 98.2%")
        
        st.markdown("---")
        st.markdown("### **Developer**")
        st.markdown("**Saif Alaswad**")
    
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>🎯 92%</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>⚡ 17,000+</h3>
                <p>Training Samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>🔬 8</h3>
                <p>Features Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>🎯 About the Model</h3>
                <p>This platform uses a <strong>Gradient Boosting Classifier</strong> trained on comprehensive patient data to predict diabetes risk with high accuracy.</p>
                <ul>
                    <li>✅ 98.2% ROC-AUC Score</li>
                    <li>✅ Balanced precision & recall</li>
                    <li>✅ Real-time predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>🔬 Key Features</h3>
                <ul>
                    <li>📊 Demographics (Age, Gender)</li>
                    <li>❤️ Medical History (Hypertension, Heart Disease)</li>
                    <li>🚬 Lifestyle (Smoking History)</li>
                    <li>⚕️ Clinical Measurements (BMI, HbA1c, Blood Glucose)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        # Performance metrics visualization
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.92, 0.92, 0.93, 0.925, 0.982]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                     color='Score',
                     color_continuous_scale='viridis',
                     range_y=[0.8, 1])
        fig.update_layout(title="Model Performance Metrics", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Predict":
        st.markdown("## 🔮 Diabetes Risk Prediction")
        st.markdown("### Enter Patient Information Below")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Demographics")
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    help="Select patient's gender"
                )
                
                age = st.slider(
                    "Age",
                    min_value=0,
                    max_value=120,
                    value=45,
                    step=1,
                    help="Enter patient's age"
                )
                
                bmi = st.number_input(
                    "BMI",
                    min_value=10.0,
                    max_value=60.0,
                    value=25.0,
                    step=0.1,
                    help="Body Mass Index"
                )
            
            with col2:
                st.markdown("#### ❤️ Medical History")
                hypertension = st.selectbox(
                    "Hypertension",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the patient have hypertension?"
                )
                
                heart_disease = st.selectbox(
                    "Heart Disease",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the patient have heart disease?"
                )
                
                smoking_history = st.selectbox(
                    "Smoking History",
                    options=["never", "No Info", "former", "current", "not current", "ever"],
                    help="Select patient's smoking history"
                )
            
            with col3:
                st.markdown("#### 🔬 Clinical Measurements")
                hba1c_level = st.slider(
                    "HbA1c Level",
                    min_value=3.5,
                    max_value=9.0,
                    value=5.5,
                    step=0.1,
                    help="HbA1c level (3.5 - 9.0)"
                )
                
                blood_glucose = st.slider(
                    "Blood Glucose Level",
                    min_value=80,
                    max_value=300,
                    value=120,
                    step=5,
                    help="Blood glucose level (80 - 300)"
                )
            submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)
        if submitted:
            input_data = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'smoking_history': smoking_history,
                'bmi': bmi,
                'HbA1c_level': hba1c_level,
                'blood_glucose_level': blood_glucose
            }
            with st.spinner("🔄 Analyzing patient data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                prediction, probability = predict_diabetes(models, input_data)
                progress_bar.empty()
            
            if prediction is not None:
                save_prediction(input_data, prediction, probability)
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                risk_level, risk_icon, risk_class = get_risk_level(probability)
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); border-radius: 10px; color: white;">
                            <h2 style="font-size: 3rem;">{risk_icon}</h2>
                            <h2>HIGH RISK</h2>
                            <p>Probability of Diabetes: {probability:.1%}</p>
                            <span class="{risk_class}">{risk_level} Risk</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); border-radius: 10px; color: white;">
                            <h2 style="font-size: 3rem;">{risk_icon}</h2>
                            <h2>LOW RISK</h2>
                            <p>Probability of Diabetes: {probability:.1%}</p>
                            <span class="{risk_class}">{risk_level} Risk</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig = create_gauge_chart(probability)
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.markdown("## 📋 Risk Factors Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_factors = []
                    if age > 50:
                        risk_factors.append(("Age > 50", age))
                    if bmi > 30:
                        risk_factors.append(("BMI > 30 (Obese)", f"{bmi:.1f}"))
                    elif bmi > 25:
                        risk_factors.append(("BMI 25-30 (Overweight)", f"{bmi:.1f}"))
                    if blood_glucose > 140:
                        risk_factors.append(("High Blood Glucose", f"{blood_glucose}"))
                    elif blood_glucose > 100:
                        risk_factors.append(("Elevated Blood Glucose", f"{blood_glucose}"))
                    if hba1c_level > 6.5:
                        risk_factors.append(("High HbA1c", f"{hba1c_level:.1f}"))
                    elif hba1c_level > 5.7:
                        risk_factors.append(("Pre-diabetic HbA1c", f"{hba1c_level:.1f}"))
                    if hypertension == 1:
                        risk_factors.append(("Hypertension", "Yes"))
                    if heart_disease == 1:
                        risk_factors.append(("Heart Disease", "Yes"))
                    if smoking_history in ['current', 'former', 'not current']:
                        risk_factors.append(("Smoking History", smoking_history))
                    if risk_factors:
                        for factor, value in risk_factors:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"• {factor}")
                            with col_b:
                                st.write(value)
                    else:
                        st.write("• No major risk factors detected")
                
                with col2:
                    st.markdown("### Risk Meter")
                    st.progress(float(probability))
                    st.caption(f"Risk Score: {probability:.1%}")
                    st.markdown("### Normal Ranges")
                    st.info("""
                    - **BMI:** 18.5 - 24.9
                    - **Blood Glucose:** < 100 mg/dL
                    - **HbA1c:** < 5.7%
                    """)
                st.markdown("---")
                st.markdown("## 💡 Recommendations")
                
                if prediction == 1:
                    st.error("""
                    ### ⚠️ High Risk Detected
                    
                    1. **Consult Healthcare Provider** - Schedule an appointment immediately
                    2. **Blood Sugar Monitoring** - Start regular glucose monitoring
                    3. **Dietary Changes** - Reduce sugar and carbohydrate intake
                    4. **Exercise** - Aim for 30 minutes of daily physical activity
                    """)
                else:
                    st.success("""
                    ### ✅ Low Risk - Keep It Up!
                    
                    1. **Maintain Healthy Diet** - Continue balanced eating habits
                    2. **Regular Exercise** - Stay active with 150 minutes/week
                    3. **Annual Check-ups** - Regular health screenings
                    4. **Weight Management** - Maintain healthy BMI
                    """)
                st.markdown("---")
                st.caption("""
                ⚕️ **Medical Disclaimer:** This prediction is for informational purposes only. 
                Always consult with a qualified healthcare provider for medical advice.
                """)
    
    elif page == "📊 History":
        st.markdown("## 📊 Prediction History")
        
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(history_df))
            with col2:
                diabetic_count = len(history_df[history_df['prediction'] == 'Diabetic'])
                st.metric("High Risk Cases", diabetic_count)
            with col3:
                st.metric("Average Age", f"{history_df['age'].mean():.1f}")
            with col4:
                st.metric("Average BMI", f"{history_df['bmi'].mean():.1f}")
            st.markdown("### 📋 Recent Predictions")
            st.dataframe(
                history_df.sort_values('timestamp', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            if st.button("📥 Export History to CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No prediction history yet. Make some predictions to see them here!")
    
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This Platform")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🏥 Diabetes Risk Assessment Platform
            This advanced medical AI platform leverages machine learning 
            to provide accurate diabetes risk predictions. Built with Streamlit and powered 
            by a Gradient Boosting model.
            
            #### 🎯 Key Features
            - **Real-time Predictions** - Instant results with high accuracy
            - **Comprehensive Analysis** - 8 key risk factors analyzed
            - **Visual Insights** - Interactive charts and gauges
            - **History Tracking** - Save and export predictions
            - **Personalized Recommendations** - Custom advice based on results
            #### 🔬 Model Details
            - **Algorithm:** Gradient Boosting Classifier
            - **Accuracy:** 92%
            - **ROC-AUC:** 98.2%
            - **Training Data:** 17,000+ patient records
            #### 📊 Features Used
            - Demographics (Age, Gender)
            - Medical History (Hypertension, Heart Disease)
            - Lifestyle (Smoking History)
            - Clinical Measurements (BMI, HbA1c, Blood Glucose)
            """)
        with col2:
            st.image("https://img.icons8.com/color/200/000000/artificial-intelligence.png")
            st.markdown("""
            ### 👨‍💻 Developer
            **Saif Alaswad**  
            **Email:** saifalaswad43@gmail.com  
            **GitHub:** [@saifalaswad43](https://github.com/saifalaswad43)
            ### 📚 References
            - CDC Diabetes Statistics
            - WHO Diabetes Guidelines
            """)
if __name__ == "__main__":
    main()