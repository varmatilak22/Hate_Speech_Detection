from data_preprocessing import load_data,preprocessing,text_preprocessing
from model_training import train_model
from model_evaluation import evaluate
import os 

def run_pipeline():
    """
    Pipeline: It is the pipeline for the Machine learning project
    1.Data Extraction
    2.Data preprocessing
    3.Model Training
    4.Model Evaluation
    5. Repeat until satisified
    """
    X=load_data()
    X_data=preprocessing(X)
    X_train,X_test,y_train,y_test=text_preprocessing(X_data)
    train_model(X_train,y_train)
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'model')
    file_path=os.path.join(out_dir_2,'decisiontree.pkl')
    cv_path=os.path.join(out_dir_2,'countvectorizer.pkl')
    evaluate(file_path,cv_path,X_test,y_test)

if __name__=='__main__':
    run_pipeline()