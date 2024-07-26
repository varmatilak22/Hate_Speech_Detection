import streamlit as st
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import clean_data
import os

##Load the model
current_dir=os.path.dirname(__file__)
out_dir=os.path.dirname(current_dir)
out_dir_2=os.path.join(out_dir,'model')
file_path=os.path.join(out_dir_2,'decisiontree.pkl')
cv_path=os.path.join(out_dir_2,'countvectorizer.pkl')

model=joblib.load(file_path)
cv=joblib.load(cv_path)

def predict(text):
    preprocess_text=clean_data(text)
    vectorized_text=cv.transform([preprocess_text])
    prediction=model.predict(vectorized_text)
    return prediction[0]

#Streamlit UI
st.title("Hate Speech Recognition")
st.write("Enter your text below:")

input_text=st.text_area("Input Text","")

if st.button('perfect'):
    if input_text:
        prediction=predict(input_text)
        st.write(f"Predicted Class:{prediction}")
    else:
        st.write("Please enter some text")