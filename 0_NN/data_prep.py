# data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def prepare_data(file_path):
    my_df = pd.read_csv(file_path)
    my_df['species'] = my_df['species'].replace(['setosa', 'versicolor', 'virginica'], [0.0, 1.0, 2.0])
    
    X = my_df.drop('species', axis=1).values
    y = my_df['species'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test
