import pandas as pd

def load_data():
    df = pd.read_csv('data/wine.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y
