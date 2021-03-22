import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import Normalizer
import yaml

df = pd.read_csv('haberman_processed.csv', index_col=0)

y = df.pop('class').to_numpy()

X = df.to_numpy()

X = Normalizer().fit_transform(X)

clf = LogisticRegression(solver=yaml.safe_load(open('params.yaml'))['solver'])
y_pred = cross_val_predict(clf, X, y, cv = yaml.safe_load(open('params.yaml'))['cv'])

acc = np.mean(y_pred==y)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)