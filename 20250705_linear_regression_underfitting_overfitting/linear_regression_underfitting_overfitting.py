import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error # 均方差
from sklearn.model_selection import train_test_split

def underfitting_overfitting():
    # 1.准备数据x, y，增加上噪声
    np.random.seed(666)  
    x = np.random.uniform(-3, 3, size=100) # uniform: 均匀分布
    y = 0.5*x**2 + x + 2 + np.random.normal(0, 1, size=100) # normal: 正态分布

    # 2.训练模型
    # 2.1 实例化线性回归模型
    estimator= LinearRegression()
    # 2.2 模型训练
    # 2.2.1 underfitting
    X  = x.reshape(-1, 1)  # 将一维数组转换为二维数组
    estimator.fit(X, y)

    # 2.2.2 good fit
    # X  = x.reshape(-1, 1)  # 将一维数组转换为二维数组
    # X2 = np.hstack([X, X**2])  # 添加二次项
    # estimator.fit(X, y)

    # 2.2.2 overfitting
    # X  = x.reshape(-1, 1)  # 将一维数组转换为二维数组
    # X2 = np.hstack([X, X**2, X**3, X**4, X**5, X**6])  # 添加多项式项
    # estimator.fit(X2, y)

    # 3.模型预测
    y_predict = estimator.predict(X)

    # 4.模型评估，计算均方差
    # 4.1 模型评估MSE
    myret = mean_squared_error(y, y_predict)    
    print('myret->', myret)
    # 4.2可视化
    plt.scatter(x, y)
    plt.plot(x, y_predict, color='r')
    plt.show(block=True)   