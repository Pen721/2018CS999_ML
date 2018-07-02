import numpy as np
import matplotlib
from sklearn import datasets
import math
from scipy.special import expit as sigmoid
import gzip
import pickle

with gzip.open('mnist.pkl.gz', 'rb') as f:
    #train_set, valid_set, test_set = pickle.load(f)

def initialize(input, hidden, output, examples):

    print("examples", examples, "input", input, "hidden", hidden, "output", output)

    x = np.random.uniform(0, 1, (examples, input))

    y = np.random.uniform(0, 1, (examples, output))

    weightsin = np.random.uniform(-1, 1, size=(input, hidden))

    weights2 = np.random.uniform(-1,1, size = (hidden+1, hidden))

    weightsout = np.random.uniform(-1, 1, size=(hidden+1, output))

    error = 1
    while(error !=0):
        #forward propogation
        layer1 = sigmoid(np.dot(x, weightsin))
        layer1 = np.append(layer1, np.ones((examples, 1)), axis = 1)

        layer2 = sigmoid(np.dot(layer1,weights2))
        layer2 = np.append(layer2, np.ones((examples, 1)), axis = 1)

        predict = sigmoid(np.dot(layer2, weightsout))
        alpha = 0.006

        error = 1/2 * (y - predict) ** 2
        error = error.sum()

        #backward propogation
        change3 = -1 * (y - predict) * dsigmoid(predict)
        DJWo = np.dot(layer2.transpose(), change3)
        weightsout = weightsout + -alpha * DJWo

        change2 = np.dot(change3, weightsout[0:np.shape(weightsout)[0]-1].transpose()) * dsigmoid(layer2[:, 0:np.shape(layer2)[1]-1])
        DJW2 = np.dot(layer1.transpose(), change2)
        weights2 = weights2 + -alpha*DJW2

        change1 = np.dot(change2, weights2[0:np.shape(weights2)[0]-1, :].transpose())*dsigmoid(layer1[:, 0:np.shape(layer1)[1]-1])
        DjWi = np.dot(x.transpose(), change1)
        weightsin += -alpha*DjWi
        print(error)

    print("Final Error", error)



def dsigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z


initialize(1,200, 1, 10)







