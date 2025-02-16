from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

randClf = RandomForestClassifier(random_state=42)
randClf.fit(X_train, y_train)

extraClf = ExtraTreesClassifier(random_state=42)
extraClf.fit(X_train, y_train)

y_rand = randClf.predict(X_test)
y_extra = extraClf.predict(X_test)

print("Accuracy Score Random Forests {}".format(accuracy_score(y_test, y_rand)))
print("Accuracy Score Extra Trees {}".format(accuracy_score(y_test, y_extra)))