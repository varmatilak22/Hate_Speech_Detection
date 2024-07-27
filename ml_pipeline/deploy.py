import streamlit as st
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import clean_data
import os

st.set_page_config(page_title='Hate Speech Recognition', layout='wide')

# Load the model and Count vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """
    Loading the model and Count vectorizer for the input text
    """
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.dirname(current_dir)
    out_dir_2 = os.path.join(out_dir, 'model')
    file_path = os.path.join(out_dir_2, 'decisiontree.pkl')
    cv_path = os.path.join(out_dir_2, 'countvectorizer.pkl')
    model = joblib.load(file_path) 
    cv = joblib.load(cv_path)
    return model, cv, file_path, cv_path

model, cv, file_path, cv_path = load_model_and_vectorizer()

@st.cache_data
def predict(text):
    preprocess_text = clean_data(text)
    vectorized_text = cv.transform([preprocess_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Custom CSS for navbar title
st.markdown("""
<style>
     .head{
     display:flex;
     justify-content:center;
     align-items:center;
     height:80px;
     background-color:#f0f0f0;
     border-bottom:1px solid #ddd;
     position:sticky;
     top:0;
     z-index:2;
     }
     .head h1{
     margin:0;
     font-size:40px;
     color:#333;
     }
     .stApp{
     padding-top:0;
     }
</style>
<div class='head'>
<h1>Hate Speech Detection</h1>
</div>
""", unsafe_allow_html=True)

st.write("""
Enter your text below to classify it into one of the following categories:
- **0:** Neither Hate Speech nor Offensive Language
- **1:** Hate Speech
- **2:** Offensive Language
""")

input_text = st.text_area("Input Text", "")

if st.button('Classify'):
    if input_text:
        prediction = predict(input_text)
        if prediction == 0:
            st.success("**Neither Hate Speech Nor Offensive Language**")
        elif prediction == 1:
            st.error("The text is classified as **Hate Speech**")
        elif prediction == 2:
            st.warning("The text is classified as **Offensive Language**")
    else:
        st.write("Please enter some text")

st.write("""
### About the Model
This model uses a Decision Tree algorithm to classify text into predefined categories based on its content. The Count Vectorizer transforms the text into a format suitable for the model. The categories are defined as follows:
- **0:** Text that is neither hateful nor offensive.
- **1:** Text that contains hate speech.
- **2:** Text that is offensive but not necessarily hateful.

**Model Accuracy:** Approximately 88%.
""")

# Add explanation of Decision Tree algorithm
st.write("""
### How the Decision Tree Algorithm Works

A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It splits the data into subsets based on the feature values, creating a tree-like model of decisions. The tree consists of nodes where each node represents a decision rule based on a feature, and branches represent the outcome of the decision. 

- **Root Node:** Represents the entire dataset and splits into two or more sub-nodes.
- **Decision Nodes:** Represent the feature-based decisions.
- **Leaf Nodes:** Represent the final output or class labels.

The algorithm continues splitting the nodes until it reaches the maximum depth or all data points belong to the same class.
""")

# Add explanation of NLP preprocessing steps
st.write("""
### NLP Preprocessing Steps

1. **Stemming:** The process of reducing words to their base or root form. For example, "running" becomes "run". This helps in reducing the dimensionality of the data.

2. **Stop Words Removal:** Stop words are common words that are typically filtered out during text processing. Examples include "and", "the", "is", etc. Removing stop words helps to focus on meaningful words in the text.

3. **Count Vectorizer:** A technique to convert text data into numerical data. It counts the occurrence of each word in the document and represents the text as a matrix of token counts. This matrix is then used as input to the model for training and prediction.
""")

# Add evaluation metrics
st.write("### Evaluation Metrics")

# Display classification report
st.write("##### Classification Report")

report_path=os.path.join(os.path.dirname(__file__),'classification_report.png')
st.image(report_path)

# Display Confusion Matrix
st.write("##### Confusion Matrix")
confu_path=os.path.join(os.path.dirname(__file__),'confusion_matrix.png')
st.image(confu_path)

# Add social media links
st.write('### Connect with Me')
st.write("[LinkedIn](https://www.linkedin.com/in/varmatilak) | [GitHub](https://github.com/varmatilak22) | [Kaggle](https://www.kaggle.com/xenowing)")

st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f0;
        }
        .css-18e3th9 {
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)