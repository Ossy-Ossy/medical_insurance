import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.write("""
# Welcome To The AI-Powered Medical Insurance Cost Predictor App

This app predicts the insurance cost estimate of an individual based on determinant factors. 
""")

st.write('***')

scaler = joblib.load("C:\\Users\\hp\\Downloads\\scaler_medical_insurance.joblib")
model = joblib.load("C:\\Users\\hp\\Downloads\\model_medical_insurance.joblib")
x = pd.read_csv("C:\\Users\\hp\\Downloads\\x_medical_insurance.csv")
x.drop(x.columns[0],axis = 1,inplace = True)

df = pd.read_csv("C:\\Users\\hp\\OneDrive\\Medical_Insurance.csv")
# Strip whitespace from column names
df.columns = df.columns.str.strip()

def insurance_pred(x,model,name,scaler,age,bmi,children,gender,smoker,location):
    fv = np.zeros(len(x.columns))
    age,bmi,children = float(age) ,float(bmi) ,float(children)
    scaled_values = scaler.transform(np.array([[age,bmi,children]]))[0]
    fv[0:3] = scaled_values
    obj_cols = [gender ,smoker ,location]
    for cols in obj_cols:
        index = np.where(x.columns == cols)[0]
        if index.size > 0:
            fv[index] = 1
    fv = fv.reshape(1,-1)
    prediction = model.predict(fv)[0]
    prediction = np.expm1(prediction)
    prediction = prediction.round()
    return prediction

st.subheader("Carefully Input Individual Information to determine Cost Estimate")

name_ch = st.text_input("What is your name?")

gender_ch = st.selectbox("Select your sex",df['sex'].unique())


age_ch = st.number_input("What is your age?" ,
                      min_value = df['age'].min(),
                      max_value = df['age'].max()
                      )
bmi_ch = st.number_input("What is your BMI (Body mass index)??",
                         min_value = df['bmi'].min(),
                         max_value = df['bmi'].max()
                         )

st.write('***')
smoker_ch = st.selectbox("Do you smoke drugs of any kind?, smokers are charged higher costs" ,df['smoker'].unique())

location_ch = st.selectbox("What region of the states are you located in??" ,df['region'].unique())
children_ch = st.number_input("How many dependents/Children does this Insurance Cover??",
                              min_value = df['children'].min(),
                              max_value = df['children'].max()
                              )
st.sidebar.header("📝 About BMI")
st.sidebar.write("""
**BMI Categories:**
- Underweight: Below 18.5
- Normal: 18.5 - 24.9
- Overweight: 25.0 - 29.9  
- Obese: 30.0 and above
""")

st.sidebar.header("ℹ️ How it works")
st.sidebar.write("""
This app uses machine learning to predict insurance costs based on:
- Age
- BMI (Body Mass Index)
- Number of children/dependents
- Gender
- Smoking status
- Geographic region
""")

if st.button("Insurance Cost"):
    price = insurance_pred(x,model,name_ch ,scaler ,age_ch,bmi_ch,children_ch,gender_ch,smoker_ch,location_ch)
    st.success(f"{name_ch}, Your Medical Insurance fee is estimated to be ${price}")
