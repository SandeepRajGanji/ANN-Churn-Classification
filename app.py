import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from tensorflow.keras.models import load_model

st.title("Churn Predication using ANN")

model = load_model('model.keras')
with open('scalar.pkl','rb') as f:
    scalar = pickle.load(f)
with open('onehot_encoder_geo.pkl','rb') as f:
   onehot_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)



CreditScore = st.number_input("CreditScore",value=None,placeholder="Type a number...")
Geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender",label_encoder_gender.classes_)
Age = st.slider("Age",min_value=18)
Tenure = st.slider("Tenure",min_value=1)
Balance = st.number_input("Balance",value=None,placeholder="Type a number...")
NumOfProducts = st.number_input("NumOfProducts",min_value=1,max_value=4,placeholder="Type a number...")
HasCrCard = st.selectbox("HasCrCard",[0,1])
IsActiveMember = st.selectbox("IsActiveMember",[0,1])
EstimatedSalary = st.number_input("EstimatedSalary",value=None,placeholder="Type a number...")

input_data = {
    'CreditScore':CreditScore,
    'Geography':Geography,
    'Gender' : Gender,
    'Age' : Age,
    'Tenure' : Tenure,
    'Balance':Balance,
    'NumOfProducts':NumOfProducts,
    'HasCrCard':HasCrCard,
    'IsActiveMember':IsActiveMember,
    'EstimatedSalary':EstimatedSalary
}

input_df = pd.DataFrame([input_data])

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
geo_encoder = onehot_encoder_geo.transform(input_df[['Geography']])

geo_data = pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = input_df.drop('Geography',axis=1)
input_df = pd.concat([input_df,geo_data],axis=1)
input_scaled = scalar.transform(input_df)

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]
st.write(f'Churn Probability :{prediction_prob:.2f}%')
if prediction_prob > 0.5:
  st.write('The customer is likely to churn.')
else :
  st.write('The customer is not likely to churn.')