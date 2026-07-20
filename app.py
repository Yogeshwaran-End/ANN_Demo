import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd 
import pickle



# Model loading is done here
model = tf.keras.models.load_model('model.h5')

# load the pickle

with open('one_hot_encoder_geo.pkl','rb') as file:
    oh = pickle.load(file)

with open('lable_encode_gender.pkl','rb') as file:
    le = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

st.title("customer churn prediction")


# geography = st.selectbox('Geography',oh.categories_[0])


# User input
geography = st.selectbox('Geography', oh.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [le.transform([gender]) [0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})


geo = oh.transform([[geography]]).toarray()
geo = pd.DataFrame(geo,columns=oh.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data,geo],axis=1)


scal = scalar.transform(input_data)


prediction = model.predict(scal)

prediction_probility = prediction[0][0]

if prediction_probility > 0.5:
    st.write("the customer are likly to charn")
else:
    st.write("the customer are not likely charn")
