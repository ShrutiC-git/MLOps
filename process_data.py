import pandas as pd
import numpy as np
from sklearn import preprocessing

filename = 'haberman.csv'
columns = ['age', 'year', 'node', 'class']
df = pd.read_csv(filename, header=None, names=columns)
labelEncoder = preprocessing.LabelEncoder()
df['class'] = labelEncoder.fit_transform(df['class'])
df.to_csv("haberman_processed.csv")