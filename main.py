from sklearn.datasets import load_breast_cancer
from DecisionTrees.DecisionTrees import DecisionTree
# from Ensemble.RandomForest import RandomForest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

randForest = DecisionTree()
randForest.fit(X_train, y_train)

y_pred = randForest.predict(X_test)

print("Accuracy score => {}".format(accuracy_score(y_test, y_pred)))