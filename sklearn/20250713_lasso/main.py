import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def l1_regularization():
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # # train the model    
    # # Lasso/L1 regularization regularization model, alpha is the regularization strength
    # # the larger the alpha, the stronger the regularization
    # # normalize=True means to normalize the features
    # estimator = Lasso(alpha=0.005, normalize=True)

    # Create a pipeline that first scales the data then applies Lasso
    # StandardScaler is used to standardize the features by removing the mean and scaling to unit variance, to make the fearture values comparable, which between 0 and 1
    # Lasso is used for L1 regularization
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.01)) # You may need to adjust alpha
    ])

    # fit the model
    X = x.reshape(-1, 1)  # Reshape x to be a 2D array
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    pipeline.fit(X3, y)
    print("Coefficients:", pipeline.named_steps['lasso'].coef_)

    y_predict = pipeline.predict(X3)

    myret = mean_squared_error(y, y_predict)
    print("Mean Squared Error:", myret)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red', linewidth=2)

    plt.show()


l1_regularization()