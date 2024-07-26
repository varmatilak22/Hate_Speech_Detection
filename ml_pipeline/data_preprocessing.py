import sqlite3
import pandas as pd
import re
import os
import string
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    """
    Load the data from database using sqlite3 and sql queries
    """
    
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'data')
    file_path=os.path.join(out_dir_2,'tweets.db.')
    print(file_path)
    conn=sqlite3.connect(file_path)
    cursor=conn.cursor()

    cursor.execute("select * from tweets;")
    data=cursor.fetchall()
    conn.close()
    return data

def preprocessing(data):
    """
    Data Preprocessing function preprocess the data
    It will have only relevant columns in it.
    """
    df=pd.DataFrame(data)
    #print(df)
    df.rename(columns={0:'count',1:'hate_speech',2:'offensive_langauge',3:'neither',4:'class',5:'tweet'},inplace=True)
    #print(df.columns)

    print("Dataset Infromation:")
    df.info()

    print("Dataset Desribe")
    #print(df.describe())

    #Data Transformation
    df['labels']=df['class'].map({0:"hate_speech",1:'offensive_language',2:'Neither'})
    #print(df)

    dataset=df[['tweet','labels']]
    #print(dataset)
    return dataset

def text_preprocessing(data):
    """
    Text preprocessing in NLP
    we are using preprocessing steps like 
    1.Stopwords removal
    2.Stemming
    3.Lemmatization 
    4.Text Vectorization
    5.train_test_split
    """
    
    data['tweet']=data['tweet'].apply(clean_data)
    #print(data['tweet'])
    #print(data['labels'])

    X=np.array(data['tweet'])
    y=np.array(data['labels'])
    
    #0=Neither,1=Hate_speech 2=offensive language
    le=LabelEncoder()
    y=le.fit_transform(y)
    #print(y)

    cv=CountVectorizer()
    #print(X)
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    print(X_train.shape[0],len(y_train),X_test.shape[0],len(y_test))
    X_train_vectorizer=cv.fit_transform(X_train)
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'model')
    file_path=os.path.join(out_dir_2,'countvectorizer.pkl')
    joblib.dump(cv,file_path)
    return X_train_vectorizer,X_test,y_train,y_test




def clean_data(text):
    """
    Clean the text data
    """
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'data')
    file_path=os.path.join(out_dir_2,'corpora')
    nltk.data.path.append(file_path)
    #Removal of stopwords
    from nltk.corpus import stopwords
    stopwords=set(stopwords.words('english'))
    #print(stopwords)

    #Importing stemming
    stemmer=nltk.SnowballStemmer('english')
    #print(stemmer)

    text=str(text).lower()
    text=re.sub('https?:?://\s|www\.$',"",text)
    #print(text)
    text=re.sub("\[.*?\]","",text)
    #print(text)

    text=re.sub('<.*?>','',text)
    #print(text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    #print(text)
    text=re.sub('\n','',text)
    #print(text)
    
    text=re.sub('\w*\d\w*','',text)
    #print(text)
    
    text=[word for word in text.split(" ") if word not in stopwords]
    #print(text)
    text=" ".join(text)
    #print(text)

    #stemming the text
    text=[stemmer.stem(word) for word in text.split(" ")]
    #print(text)
    text=" ".join(text)
    #print(text)
    return text


if __name__=='__main__':
    X=load_data()
    X_data=preprocessing(X)
    X_train,X_test,y_train,y_test=text_preprocessing(X_data)