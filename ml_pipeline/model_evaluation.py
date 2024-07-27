from sklearn.metrics import confusion_matrix,classification_report
from data_preprocessing import load_data,preprocessing,text_preprocessing
import joblib
import os

def evaluate(model_path,cv_path,X_test,y_test):
    """
    This functions load the train model on text dataset
    """
    model=joblib.load(model_path)
    cv=joblib.load(cv_path)
    X_test_vectorizer=cv.transform(X_test)
    predict=model.predict(X_test_vectorizer)

    consfu_mat=confusion_matrix(y_test,predict)
    print(consfu_mat)

    report=classification_report(y_test,predict,output_dict=True)
    print(report)
    return consfu_mat,report


if __name__=='__main__':
    X=load_data()
    X_data=preprocessing(X)
    X_train,X_test,y_train,y_test=text_preprocessing(X_data)
    
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'model')
    file_path=os.path.join(out_dir_2,'decisiontree.pkl')
    cv_path=os.path.join(out_dir_2,'countvectorizer.pkl')
    print(file_path)
    consfu_mat,report=evaluate(file_path,cv_path,X_test,y_test)