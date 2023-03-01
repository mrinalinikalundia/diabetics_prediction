# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:17:48 2023

@author: USER
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('C:/Users/USER/Documents/GitHub/diabetics_prediction/trained_model.sav','rb'))

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    st.title('Diabetes Prediction Web App')
    
    
    Pregnancies=st.text_input('No of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BP value')
    SkinThickness=st.text_input('Skin Thickness value')
    Insulin=st.text_input('Insulin value')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age of Person')
    
    
    diagnosis=''
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()