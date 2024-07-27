import os

def absolute_path():
    """
    retruns the absolute path on countvectorizer and model
    """
    #Paths
    current_dir=os.path.dirname(__file__)
    out_dir=os.path.dirname(current_dir)
    out_dir_2=os.path.join(out_dir,'model')
    file_path=os.path.join(out_dir_2,'decisiontree.pkl')
    cv_path=os.path.join(out_dir_2,'countvectorizer.pkl')
    return file_path,cv_path