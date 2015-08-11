__author__ = 'raza'

import numpy as np
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression


np.random.seed(43)

data = np.loadtxt('simulation_data.txt', skiprows=1, delimiter=',')
np.random.shuffle(data)

'''
train_size = 10000
test_size = 1000
# training-test set split, ~65-35
x_train = data[:train_size, :-1]
y_train = data[:train_size, -1]
x_test = data[train_size: train_size + test_size, :-1]
y_test = data[train_size: train_size + test_size, -1]
'''

size = 10000
C = 10.0
gamma = 0.0
algo = 'rbfSVR'
x_train = data[:size, :-1]
y_train = data[:size, -1]
# estimator = LinearRegression()
estimator = svm.SVR(C=C, kernel='rbf', gamma=gamma)
svr_rbf = estimator.fit(x_train, y_train).predict(x_train)
print estimator.decision_function(x_train)
'''
print "support_vectors_: "
print estimator.support_vectors_.shape
print estimator.support_vectors_
print "dual_coef_: "
print estimator.dual_coef_.shape
print estimator.dual_coef_
print "intercept_: "
print estimator.intercept_.shape
print estimator.intercept_
'''
'''
file_name = "./results/training_size={0}, C={1}, folds={2}, algo={3}.txt".format(size, C, folds, algo)
with open(file_name, 'wb') as f:
    f.write("Accuracies: " + str(accs) + "\n")
    f.write("Mean: " + str(accs.mean()))

print "training_size={0}, C={1}, folds={2}, algo={3}, gamma={4}.txt".format(size, C, folds, algo, gamma)
print "Accuracies: " + str(accs) + "\n"
print "Mean: " + str(accs.mean())
'''