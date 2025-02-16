import numpy as np
import matplotlib.pyplot as plt

# => REGRESSION LINEAIRE

# ŷ = theta0 + theta1 * x1 + ... + thetai * xi
"""
1. ajouter le theta0 à X
2. calculer les valeurs de theta en sachant que
   theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
3. faire les predictions avec X_new.dot(theta)
"""

def train(X, y):
    line, column = X.shape
    X_b = np.c_[np.ones((line, column)), X]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def predict(X, model):
    line, column = X.shape
    data_b = np.c_[np.ones((line, column)), X]
    return data_b.dot(model)

def afficher(X, y, y_pred):
    plt.plot(X, y_pred, "r-")
    plt.plot(X, y, "b.")
    plt.show()


# ==> EXEMPLE
X = 5 * np.random.randn(100, 1)
y = 2.5 * X + np.random.randn(100, 1)

model = train(X, y)
y_predicted = predict(X, model)
afficher(X, y, y_predicted)
