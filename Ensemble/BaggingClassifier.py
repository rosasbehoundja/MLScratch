from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bagClf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, oob_score=True)

bagClf.fit(X_train, y_train)
y_pred = bagClf.predict(X_test)

print("Out of bag Score => {}".format(bagClf.oob_score_))
print("Accuracy score => {}".format(accuracy_score(y_true=y_test, y_pred=y_pred)))