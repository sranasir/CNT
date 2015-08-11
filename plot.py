__author__ = 'raza'

import numpy as np
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(43)

data = np.loadtxt('simulation_data.txt', skiprows=1, delimiter=',')
np.random.shuffle(data)



size = 10000
C = 10.0
gamma = 0.0
algo = 'Linear Regression'
x_train = data[:size, :-1]
y_train = data[:size, -1]
# estimator = svm.SVR(C=C, kernel='rbf', gamma=gamma)

# estimator = LinearRegression()
estimator = svm.SVR(C=C, kernel='rbf', gamma=gamma)
y_pred = estimator.fit(x_train, y_train).predict(x_train)

# 3D plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c='k', label='data')
plt.hold('on')
plt.plot(x_train[:, 0], x_train[:, 1], y_pred, c='g', label='RBF Model', linestyle=':')
ax.set_xlabel('# of tubes')
ax.set_ylabel('ratio')
ax.set_zlabel('conductance')
plt.title('SVR')
plt.legend()
plt.show()



# 2D plot
'''
plt.scatter(x_train[:, 0], y_train, c='k', label='data')
plt.hold('on')
# plt.plot(x_train, y_pred, c='g', label=algo)
plt.xlabel('# of tubes')
plt.ylabel('conductance')
plt.title('SVR')
plt.legend()
plt.show()
'''



