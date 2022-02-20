from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def get_data() -> Tuple[pd.DataFrame,pd.DataFrame]:
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    return (train_df, test_df)

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    y = df.Survived
    X = df.drop(columns=['Survived'])
    return (X, y)

def process_data(x: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    dummy_gender = pd.get_dummies(x['Sex'])
    dummy_embarked = pd.get_dummies(x['Embarked'])
    x_processed = x.drop(columns=['PassengerId', 'Sex' ,'Name',  'Ticket', 'Cabin', 'Embarked'])
    x_processed = pd.merge(
        left=x_processed, 
        right= dummy_gender,
        left_index=True,
        right_index=True
    )
    x_processed = pd.merge(
        left=x_processed, 
        right= dummy_embarked,
        left_index=True,
        right_index=True
    )
    x_processed['Age'].fillna(x_processed['Age'].mean(), inplace=True)
    x_processed['Fare'].fillna(x_processed['Fare'].mean(), inplace=True)
    x_processed['Pclass'].fillna(x_processed['Pclass'].median(), inplace=True)
    return x_processed, x['PassengerId']

def get_best_model(x,y):
    knn_clf = RandomForestClassifier()
    params_grid = {
        'n_estimators': [*range(30,101)],
        'n_jobs': [-1],
    }
    grid_clf = GridSearchCV(knn_clf, params_grid, cv=6)
    grid_clf.fit(x, y)
    print(grid_clf.best_params_)
    return grid_clf.best_estimator_

def get_predictions(model, x):
    y_hat = model.predict(x)
    return y_hat

if __name__ == '__main__':
    train_df, test_df = get_data()
    X, y = split_data(train_df)
    x_processed = process_data(X)[0]
    best_model = get_best_model(x_processed, y)
    # print(best_model)
    x_test, ids = process_data(test_df)
    predictions = get_predictions(best_model, x_test)
    # print(type(predictions))
    # print(predictions)
    response_df = pd.DataFrame()
    response_df['PassengerId'] = ids
    response_df['Survived'] = predictions
    response_df.to_csv('response.csv', index=False)
    
