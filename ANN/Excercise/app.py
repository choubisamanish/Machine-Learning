import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import pickle
import os

model = tf.keras.models.load_model('./ANN/churn_model.h5')

# load encoder and scaler
path = './ANN/label_encodergender.pkl'
with open(path, 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('./ANN/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./ANN/one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

## streamlit app
st.title( "Customer Churn Prediction")
geography = st.selectbox( 'Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider( 'Age', 18, 92)
balance = st.number_input('Balance')
Credit_Score = st.number_input('Credit Score')
EstimatedSalary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0, 10)
num_of_Products = st.slider('Number of Products', 1 ,4)
HasCrCard = st.selectbox('Has Credit Card',[0,1])
IsActiveMember = st.selectbox("Is Active Member",[0,1])

#prepare input data
input_data = pd.DataFrame(
    {
        'CreditScore':[Credit_Score],
        'Gender' : [label_encoder_gender.transform([gender])[0]],
        'Age' : [age],
        'Tenure':[tenure],
        'Balance' : [balance],
        'NumOfProducts':[num_of_Products],
        'HasCrCard' : [HasCrCard],
        'IsActiveMember' : [IsActiveMember],
        'EstimatedSalary':[EstimatedSalary]
    }
)

#one hot encoding
geo_enecoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_enecoded_def = pd.DataFrame(geo_enecoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

#combine one hot encoding geo dat awith input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_enecoded_def], axis=1)

#scale input data
input_data_scale = scaler.transform(input_data)

#predict Churn
prediction = model.predict(input_data_scale)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability:{prediction_prob:.2f}')

if prediction_prob > 0.5 :
    st.write('The Customer is likely to churn')
else:
    st.write('The Customer is not likely to churn')
