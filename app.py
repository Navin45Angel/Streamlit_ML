#import necessary libraries
import streamlit as st
import numpy as np
import pickle

# Loaded my trained model which i made in stroke.ipynb and saved as 'strokemodel.pkl'
with open('strokemodel.pkl', 'rb') as f:
    model = pickle.load(f)


# Set up the Streamlit app
# Streamlit configuration like page title, icon, and layout
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧑‍⚕️", layout="centered")
st.title("🧑‍⚕️ Stroke Risk Prediction App")  #icon were downloaded from the web
# Custom CSS to enhance the appearance of the app
st.markdown(
    """
    <style>
    .stRadio > label {font-size: 18px;}
    .stSelectbox > label {font-size: 18px;}
    .stSlider > label {font-size: 18px;}
    </style>
    """, unsafe_allow_html=True
)
# Introduction text
st.write("Enter your details below to predict your risk of stroke:")

# Dictionaries for mapping
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
married_map = {'Yes': 1, 'No': 0}
work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
residence_map = {'Urban': 0, 'Rural': 1}
smoking_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

# Streamlit widgets for input
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Gender", list(gender_map.keys()), horizontal=True)
    age = st.slider("Age", 0, 100, 30)
    hypertension = st.radio("Hypertension (High Blood Pressure)?", ['No', 'Yes'], horizontal=True)
    heart_disease = st.radio("Heart Disease?", ['No', 'Yes'], horizontal=True)
    ever_married = st.radio("Ever Married?", list(married_map.keys()), horizontal=True)

with col2:
    work_type = st.selectbox("Work Type", list(work_type_map.keys()))
    residence_type = st.radio("Residence Type", list(residence_map.keys()), horizontal=True)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=0.1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", list(smoking_map.keys()))

# Convert categorical inputs to numbers using your mapping
gender_num = gender_map[gender]
hypertension_num = 1 if hypertension == 'Yes' else 0
heart_disease_num = 1 if heart_disease == 'Yes' else 0
ever_married_num = married_map[ever_married]
work_type_num = work_type_map[work_type]
residence_type_num = residence_map[residence_type]
smoking_status_num = smoking_map[smoking_status]

# Prepare input for prediction
input_data = np.array([[gender_num, age, hypertension_num, heart_disease_num, ever_married_num,
                        work_type_num, residence_type_num, avg_glucose_level, bmi, smoking_status_num]])

# Prediction button
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High risk of stroke detected! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Low risk of stroke. (Probability: {prob:.2%})")

    st.markdown(
        """
        <hr>
        <small>
        <b>Note:</b> This prediction is for educational purposes only and not a substitute for professional medical advice.
        </small>
        """, unsafe_allow_html=True
    )
