import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import Normalizer
import yaml
import pickle
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('haberman_processed.csv', index_col=0)

y = df.pop('class').to_numpy()

X = df.to_numpy()

X = Normalizer().fit_transform(X)

#Using a Logistic Regression in production
#clf = LogisticRegression(solver=yaml.safe_load(open('params.yaml'))['solver'])
clf = MultinomialNB()
y_pred = cross_val_predict(clf, X, y, cv = yaml.safe_load(open('params.yaml'))['cv'])

# Metrics for comparing performance bw models
acc = np.mean(y_pred==y)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

# Dumping the model into a serialized file
model_name = "final_model.sav"
pickle.dump(clf, open(model_name, 'wb'))
