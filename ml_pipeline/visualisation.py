import os
from model_evaluation import evaluate
from data_preprocessing import load_data,preprocessing,text_preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from helper_function import absolute_path
def confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.savefig('confusion_matrix.png')

def visualise_report(report):
    report_df = pd.DataFrame(report).transpose()
    report_df=report_df.iloc[:-3]  # Exclude support row
    report_df=report_df[['precision', 'recall', 'f1-score']]
    plt.figure(figsize=(8,6))
    plt.title('Classification Report')
    sns.heatmap(report_df,annot=True,fmt='.2f',cmap='Blues')
    plt.xlabel('Scores')
    plt.ylabel('Classes')
    plt.savefig('classification_report.png')

if __name__=='__main__':
    X=load_data()
    X_data=preprocessing(X)
    X_train,X_test,y_train,y_test=text_preprocessing(X_data)
    file_path,cv_path=absolute_path()
    confu_mat,report=evaluate(file_path,cv_path,X_test,y_test)
    confusion_matrix(confu_mat)
    visualise_report(report)
