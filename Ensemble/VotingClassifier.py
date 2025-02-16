from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=200, n_features=5, n_informative=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logReg = LogisticRegression(penalty="l2", random_state=42)
svc = SVC(C=1.0, random_state=42)
randForest = RandomForestClassifier(random_state=42)

vClass = VotingClassifier(
    estimators=[("logReg", logReg), ("svc", svc), ("randForest", randForest)],
    voting="hard"
)

classifiers = [logReg , svc, randForest, vClass]
for cls in classifiers:
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print(f"> Accuracy of {cls.__class__.__name__} => {accuracy_score(y_true=y_test, y_pred=y_pred)}")