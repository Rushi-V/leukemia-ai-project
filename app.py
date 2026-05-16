import streamlit as st
import pandas as pd
import joblib

# Run with: python3 -m streamlit run app.py

model = joblib.load("leukemia_model.pkl")
le = joblib.load("leukemia_label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Leukemia Predictor")

st.subheader("Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=18)
gender = st.selectbox("Gender", ["Male", "Female"])
wbc = st.number_input("White Blood Cell Count", min_value=0.0, value=5000.0)
rbc = st.number_input("Red Blood Cell Count", min_value=0.0, value=5000.0)
bmb = st.number_input("Bone Marrow Blasts", min_value=0.0, value=50.0)

weight = st.number_input("Weight (lbs)", min_value=0.0, value=135.0)
height = st.number_input("Height (in)", min_value=1.0, value=67.0)

bmi = (weight / (height ** 2)) * 703
st.write(f"BMI: {bmi:.2f}")

country = st.selectbox("Country", ['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Norway', 'Russia', 'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Turkey', 'Uk', 'Usa'])

ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"])
urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])

ses = st.selectbox("Socioeconomic Status", ["Low", "Medium", "High"])
infhist = st.selectbox("Infection History", ["Yes", "No"])
ch_ill = st.selectbox("Chronic Illness", ["Yes", "No"])
im_dis = st.selectbox("Immune Disorders", ["Yes", "No"])
sm_stat = st.selectbox("Smoking Status", ["Yes", "No"])
al_con = st.selectbox("Alcohol Consumption", ["Yes", "No"])
genetic = st.selectbox("Genetic Mutation", ["Yes", "No"])
family = st.selectbox("Family History", ["Yes", "No"])
radiation = st.selectbox("Radiation Exposure", ["Yes", "No"])

yes_no_map = {"No": 0, "Yes": 1}
ses_map = {"Low": 0, "Medium": 1, "High": 2}

if st.button("Predict"):
    user_data = pd.DataFrame([{
        "Age": age,
        "Gender": 0 if gender == "Male" else 1,
        "WBC_Count": wbc,
        "RBC_Count": rbc,
        "Bone_Marrow_Blasts": bmb,
        "BMI": bmi,
        "Country": country,
        "Ethnicity": ethnicity,
        "Urban_Rural": urban_rural,
        "Socioeconomic_Status": ses_map[ses],
        "Infection_History": yes_no_map[infhist],
        "Chronic_Illness": yes_no_map[ch_ill],
        "Immune_Disorders": yes_no_map[im_dis],
        "Smoking_Status": yes_no_map[sm_stat],
        "Alcohol_Consumption": yes_no_map[al_con],
        "Genetic_Mutation": yes_no_map[genetic],
        "Family_History": yes_no_map[family],
        "Radiation_Exposure": yes_no_map[radiation],
    }])

    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=feature_columns, fill_value=0)

    prediction_num = model.predict(user_data)[0]
    prediction_label = le.inverse_transform([prediction_num])[0]

    risk = model.predict_proba(user_data)[0][1] * 100

    st.subheader("Prediction Result")
    st.write("Prediction:", prediction_label)
    st.write("Estimated leukemia risk:", round(risk, 2), "%")