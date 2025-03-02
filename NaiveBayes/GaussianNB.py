import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

class GaussianNB:
    def __init__(self):
        self.mean = {}    # moyenne de chaque classe
        self.stdev = {}   # écart-type de chaque classe
        self.proba_ = {}  # probabilités à priori de chaque classe
        self.classes = None # les classes de y

    def fit(self, X, y):
        # identification des classes uniques
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c] # filtrage des données

            self.mean[c] = np.mean(X_c, axis=0)
            self.stdev[c] = np.std(X_c, axis=0, ddof=1)
            self.proba_[c] = len(X_c)/len(X)

    def predict(self, X):
        # initialisation de la liste des prédictions
        y_pred = []
            
        for x in X:
            log_prob = {}
            # initialisation du calcul avec la log_prob à priori de chaque classe
            for c in self.classes:
                log_prob[c] = np.log(self.proba_[c])

            for c in self.classes:
                for i in range(len(x)):
                    mean_c = self.mean[c][i]
                    stdev_c = self.stdev[c][i]
                    # éviter les divisions par zéro
                    if stdev_c == 0:
                        stdev_c = 1e-9
                    # ajout de la vraisemblance obtenue à partir de la densité de probabilité gaussienne
                    log_prob[c] += -0.5 * np.log(2 * np.pi) - np.log(stdev_c) - (((x[i] - mean_c)**2)/(2 * stdev_c**2))

            best_class = max(log_prob, key=log_prob.get)
            y_pred.append(best_class)

        y_pred = np.array(y_pred)
        return y_pred
    
if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    print(f"My from scratch Accuracy: {accuracy:.4f}")

    # Scikit-learn Gaussian NB
    sklearn_gnb = SklearnGaussianNB()
    sklearn_gnb.fit(X_train, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_train)

    accuracy_sklearn = accuracy_score(y_train, y_pred_sklearn)
    print(f"Sklearn GaussianNB Accuracy: {accuracy_sklearn:.4f}")