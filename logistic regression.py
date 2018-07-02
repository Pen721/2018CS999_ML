import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from scipy.special import expit

cancer = datasets.load_breast_cancer()

# load data
X = cancer.data
Y = cancer.target[:,np.newaxis]
m = len(X)
n = np.shape(X)[1]

#normalize data
for i in range(np.shape(X)[1]):
    X[:, i] = (X[:, i]-np.average(X[:, i]))/np.var(X[:, i])

#append 1 in front to have intercepts
X = np.append(np.ones([len(X), 1]), X, 1)

#splitting the data into test and train, trains specifies how many training sets there are
trains = 500
Train = X[0:trains, :]
tans = Y[0:trains, :]
Test = X[trains:len(X)]
answer = Y[trains:len(X)]
theta = np.random.rand(len(Train[0]), 1)
a = 0.0000002

#define logistical regression functions
def cost(theta1, X1, Y1):
    h = expit(X1.dot(theta1))
    return np.average(Y1 * np.log(h + 0.00001) + (np.ones([len(Y1), 1]) - Y1) * np.log(1 - h + 0.00001)) * -1

#cost comparison
print("initial cost:", cost(theta, X, Y))

#perform gradientD
c = 0;
var = True
while var:
    for i in range(len(theta)):
        prediction = expit(X.dot(theta)) - Y
        theta = theta - a * np.transpose(X).dot(prediction)

    if (np.abs(cost(theta, X, Y) - c) < 10 ** -5):
        break
    c = cost(theta, X, Y)

#calculate outcome
rights = 0
outcome = np.zeros([len(X), 1])
for i in range(len(X)):
    if(X[i, :].dot(theta) >= 0.5):
        outcome[i, 0] = 1
    if(outcome[i] == Y[i]):
        rights +=1

accuracy = rights/(len(Y)) * 100

#report these numbers
print("final cost:", cost(theta, X, Y))
print("Number Correct", rights, "Number incorrect", len(Y) - rights)
print("Percent of Accuracy:", accuracy)