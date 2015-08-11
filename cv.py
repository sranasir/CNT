__author__ = 'raza'

import numpy as np
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression


def display_scores(params, scores):
    from scipy.stats import sem

    params = ", ".join("{0} = {1}".format(k, v)
                       for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    return line


def display_grid_scores(grid_scores):
    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    f = open("grid_search.txt", "wb")
    for params, mean_score, scores in grid_scores:
        f.write(display_scores(params, scores) + '\n')
    f.close()


np.random.seed(43)

data = np.loadtxt('simulation_data.txt', skiprows=1, delimiter=',')
np.random.shuffle(data)

# params = {'C': [5, 10, 20, 30], 'epsilon': [0.1, 0.2, 0.4, 0.6]}

size = 10000
C = 10.0
gamma = 0.3
algo = 'rbfSVR'
x_train = data[:size, :-1]
y_train = data[:size, -1]
# estimator = LinearRegression()
estimator = svm.SVR(C=C, kernel='rbf', gamma=gamma)
folds = 3
# x_train, x_test, y_train, y_test = cv.train_test_split(data[:size, :-1], data[:size, -1],
#                                                        test_size=0.3, random_state=0)
it = cv.KFold(x_train.shape[0], n_folds=folds, random_state=0)
accs = cv.cross_val_score(estimator, x_train, y_train, cv=it, n_jobs=-1)
# gscv = GridSearchCV(svm.SVR(), params, cv=it, verbose=1, n_jobs=-1)
# gscv.fit(x_train, y_train)
# display_grid_scores(gscv.grid_scores_)
#
# file_name = "./results/training_size={0}, C={1}, folds={2}, algo={3}.txt".format(size, C, folds, algo)
# with open(file_name, 'wb') as f:
#     f.write("Accuracies: " + str(accs) + "\n")
#     f.write("Mean: " + str(accs.mean()))

print "training_size={0}, C={1}, folds={2}, algo={3}, gamma={4}.txt".format(size, C, folds, algo, gamma)
print "Accuracies: " + str(accs) + "\n"
print "Mean: " + str(accs.mean())
