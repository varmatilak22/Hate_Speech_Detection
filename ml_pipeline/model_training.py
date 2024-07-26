from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import load_data,preprocessing,text_preprocessing
import joblib
import os
def train_model(X_train, y_train):
    """
    Train the Decision Tree classifier model.
    """
    try:
        model = DecisionTreeClassifier(random_state=1)
        model.fit(X_train, y_train)
        
        current_dir=os.path.dirname(__file__)
        out_dir=os.path.dirname(current_dir)
        out_dir_2=os.path.join(out_dir,'model')
        file_path=os.path.join(out_dir_2,'decisiontree.pkl')
        print(file_path)
        joblib.dump(model,file_path)
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")


if __name__=='__main__':
    X=load_data()
    #print(len(X))
    X_data=preprocessing(X)
    #print(len(X_data))
    X_train,X_test,y_train,y_test=text_preprocessing(X_data)
    #print(X_train.shape)
    train_model(X_train,y_train)


