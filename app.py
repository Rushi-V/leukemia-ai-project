import streamlit as st
#Run streamlit code in terminal with following line: python3 -m streamlit run app.py

st.title("Leukemia Predictor")
st.write("Welcome to our leukemia prediction tool.")

st.subheader("Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=18)
gender = st.selectbox("Gender", ["Male", "Female"])
wbc = st.number_input("White Blood Cell Count", min_value=0.0, value=5000.0)
rbc = st.number_input("Red Blood Cell Count", min_value = 0.0, value = 5000.0)
bmb = st.number_input("Bone Marrow Blasts", min_value = 0.0, value = 50.0)
weight = st.number_input("Weight (lbs)", min_value = 0.0, value = 135.0)
height = st.number_input("Height (in)", min_value = 0.0, value = 67.0)
country = st.selectbox("Country", ['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Norway', 'Russia', 'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Turkey', 'UK', 'USA'])
ses = st.selectbox("Socioeconomic Status", ["Low", "Medium", "High"])
infhist = st.selectbox("Infection History", ["yes", "no"])
ch_ill = st.selectbox("Chronic Illness", ["yes", "no"])
im_dis = st.selectbox("Immune Disorders", ["yes", "no"])

if st.button("Predict"):
    st.write("Prediction will appear here.")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"WBC Count: {wbc}")
    st.write(f"RBC Count: {rbc}")
    st.write(f"Bone Marrow Blasts: {bmb}")
    st.write(f"Weight (lbs): {weight}")
    st.write(f"Height (in): {height}")
    st.write(f"Country: {country}")
    st.write(f"Socioeconomic Status: {ses}")
    st.write(f"Infection History: {infhist}")
    st.write(f"Chronic Illness: {ch_ill}")
    st.write(f"Immune Disorders: {im_dis}")
