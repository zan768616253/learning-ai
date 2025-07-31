import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler #特征处理
from sklearn.model_selection import train_test_split #数据集划分

from sklearn.linear_model import LinearRegression #正规方程的回归模型
from sklearn.linear_model import SGDRegressor #梯度下降回归模型

from sklearn.metrics import mean_squared_error #均方误差

from sklearn.linear_model import Ridge #岭回归
from sklearn.linear_model import RidgeCV #岭回归交叉验证

# Linear Regression
def linear_model1():
    # 2 data preprocessing
    # 2.1 Load the dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    
    # sep="\s+" - split data by one or more whitespace
    # skiprows=22 - skip the first 22 rows, beacause they are headers    
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    print(f'raw_df.head()-->\n{raw_df.head()}')
    # [::2, :] - take every second row starting from the first, even rows
    # [1::2, :2] - take every second row starting from the second and only the first two columns， odd rows
    # hstack - horizontally stack arrays
    # in data, every 2 rows present features for one sample
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    print(f'data-->\n{data}')
    target = raw_df.values[1::2, 2]

    # 2.2 Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 2.3 Standardize the features
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 3 train the model
    # 3.1 Using Linear Regression
    estimator = LinearRegression()
    # 3.2 train
    estimator.fit(X_train, y_train)

    # 4 predict
    y_predict = estimator.predict(X_test)
    print("predict:", y_predict)
    print("coefficients:", estimator.coef_) # gradient
    print("intercept:", estimator.intercept_) # intercept

    # 5 Evaluate the model 
    error = mean_squared_error(y_test, y_predict)
    print("Mean Squared Error:", error)

# SGD regression
def linear_model2():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=22)

    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    estimator = SGDRegressor()
    estimator.fit(X_train, y_train)

    y_predict = estimator.predict(X_test)
    print("predict:", y_predict)
    print("coefficients:", estimator.coef_) # 模型权重系数
    print("intercept:", estimator.intercept_) # 模型的偏置

    # 5 Evaluate the model 
    error = mean_squared_error(y_test, y_predict)
    print("Mean Squared Error:", error)