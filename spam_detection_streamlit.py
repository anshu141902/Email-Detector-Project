# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:38:00 2021

@author: 91892
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import pickle
lr_pipe = pickle.load(open('lr_pipe', 'rb'))
mul_nb_pipe = pickle.load(open('mul_nb_pipe', 'rb'))
ran_pipe = pickle.load(open('ran_pipe', 'rb'))
svc_pipe = pickle.load(open('svc_pipe', 'rb'))
tree_pipe = pickle.load(open('tree_pipe', 'rb'))


st.title('Spam Detector')
st.subheader('by Anshu')
st.markdown('----')
st.balloons()


email = st.text_area('Enter the Email: ')

activities=['MultinomialNB','Logistic Regression','Random Forest Classifier','Support Vector Classifier','Decision Tree']
option=st.sidebar.selectbox('Which model would you like to use?',activities)

inputs=np.array([email])

if option=='MultinomialNB':
    model = mul_nb_pipe
    
elif option=='Logistic Regression':
    model = lr_pipe
                         
elif option=='Random Forest Classifier':
    model = ran_pipe
                               
elif option=='Support Vector Classifier':
    model = svc_pipe
                  
else:
    model = tree_pipe
    
    
def predict_Strokes(email):
                prediction=model.predict(inputs)
                return prediction
            
            
if st.button("Prediction"):
    prediction = predict_Strokes(email)
    if prediction == 1:
                 st.error('It is a Spam')
    else:
                st.success('It is not a Spam')
            
            
            
            
            
            
            