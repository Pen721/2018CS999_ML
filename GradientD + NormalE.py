import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from numpy.linalg import inv

#load data
diabetes = datasets.load_diabetes()
diabetes_Y = diabetes.target[:, np.newaxis]
diabetes_X = diabetes.data

# Append 1s infront of the dataset inputs
diabetes_X= np.append(np.ones([len(diabetes_X), 1]), diabetes_X, 1)

#initialize theta, alaph. And helpful varaibles such as m, n
n = diabetes_X.shape[1]
m = len(diabetes_X)
theta = np.random.rand(len(diabetes_X[0]), 1) * np.random.randint(1,8)
alpha = 0.05

# normalize dataset
def noramlize(X, Y):
    for i in range(1,X.shape[1]-1):
        average = np.average(X[:,i])
        range = np.max(X[:,i]) - np.min(X[:,i])
        X[:,i] = (X[:,i] - np.ones(len[X], 1)*average)/(range)

#define cost function
def cost(X, Y, theta):
    m = len(Y)
    prediction = X.dot(theta)
    return 1/(2*m)*np.sum((prediction - Y)*(prediction - Y))

#print out weights and cost for future comparison
print("initial weights:")
print(theta)
print("initial cost:", cost(diabetes_X, diabetes_Y, theta))
print()

# perform linear regression
# c is used to compare the different in costs per update
c = 0
var = True
while var:
    prediction = diabetes_X.dot(theta) - diabetes_Y

    # update thetas
    for i in range(n):
        theta[i, 0] = theta[i, 0] - alpha * 1 / m * prediction.transpose().dot(diabetes_X[:,i])

    #determine how much the cost is decreasing per run
    dif = (abs(c - cost(diabetes_X, diabetes_Y, theta)))
    if(dif < 5 * 10**-5):
        #if the decrease in cost is very small, then end the process for times sake
            break
    #determine new cost
    c = cost(diabetes_X, diabetes_Y, theta)


#print weights and cost after linear regression
print("final weights:")
print(theta)
print("final cost :", cost(diabetes_X, diabetes_Y, theta))
print()

#compare results with matrix obtained through normal equation
theta = inv(diabetes_X.transpose().dot(diabetes_X)).dot(diabetes_X.transpose()).dot(diabetes_Y)
print("theta through normal equation:")
print(theta)
print("cost: ", cost(diabetes_X, diabetes_Y, theta))
