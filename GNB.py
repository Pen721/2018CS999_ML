import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#load dataset
wine = datasets.load_wine()
X, Y, classes = (wine.data, wine.target, wine.target_names)

var = np.zeros([len(classes),np.shape(X)[1]])
average = np.zeros([len(classes),np.shape(X)[1]])
count= np.zeros([len(classes),np.shape(X)[1]])

# finding the average
for i in range(len(X)):
    clas = Y[i]
    average[clas,:] += X[i]
    count[clas,:] += 1
average =  np.divide(average,count)

# finding the variance
for i in range(len(X)):
    clas = Y[i]
    var[clas,:] += (average[clas, :] - X[i]) ** 2
var = np.divide(var,count)

# implement the Gaussian NB classification algorithm
def probability(feature):
    p = np.zeros([3,1])
    for i in range(3):
        # (count[i, 0]/len(Y)) is P of a certain class, the rest is P of X given Y
        p[i][0] = (count[i, 0]/len(Y)) * np.prod(1/(np.sqrt(2 * np.pi * var[i, :])) * np.exp((-(feature -average[i, :]) ** 2)/ (2*var[i,:])))
    return np.argmax(p)

#calculate the prediction
y_pre = [probability(X[i, :]) for i in range(len(X))]

#calculate percentage correct
accuracy = (Y == y_pre).sum()
print("A total of", accuracy, "correct out of ",len(Y))
accuracy *= 100/(len(Y))
print("Accuracy rate:", accuracy, "%")
