import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.special import expit

# Load dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target[:, np.newaxis]

# Normalize data
for i in range(len(x[1])):
    x[:, i] = (x[:, i]-np.average(x[:, i]))/np.var(x[:, i])
numTrainingSets = 1

# Split the input into training/testing sets
xTrain = x[:-numTrainingSets]
xTest = x[-numTrainingSets:]

#add a column of ones to the matrix of inputs [this will account for a constant b in a regression y = mx+b]
inp = np.ones((len(xTrain), len(xTrain[0])+1))
inp[:, -len(xTrain[0]):] = xTrain

testInp = np.ones((len(xTest), len(xTest[0])+1))
testInp[:, -len(xTest[0]):] = xTest

# Split the outputs into training/testing sets
out = y[:-numTrainingSets]
tOut = y[-numTrainingSets:]

thetas = np.random.rand(len(xTrain[0])+1, 1)

#expit is sigmoid
def cost(x, y, theta):
    return np.average(y*np.log(expit(np.dot(x, theta))+0.0001)+(np.ones([len(y), 1])-y)*np.log(1-expit(np.dot(x, theta))+0.0001))*-1

#Perform Logistic Gradient Descent
def gradientDescent(input, output, weights, alpha, iterations):
    for i in range(iterations):
        loss = output - expit(np.dot(input, weights))
        gradient = input.T.dot(loss)
        weights = weights + (alpha * gradient)
    return weights

# Classify input matrix
def predict(X, weights):
    return [1 if expit(np.dot(inp[i, :], weights)[0]) >= 0.5 else 0 for i in range(len(X))]


print("Initial Cost:")
#print(thetas)
print(cost(inp, out, thetas))

newThetas = gradientDescent(inp, out, thetas, 0.00001, 100000)
print("Final Cost:")
print(newThetas)
print(cost(inp, out, newThetas))

correct = (cancer.target[:-numTrainingSets] == predict(inp, newThetas)).sum()
incorrect = (cancer.target[:-numTrainingSets] != predict(inp, newThetas)).sum()
accuracy = correct/len(inp)*100
print("\nCorrect: %d" % correct)
print("Incorrect: %d" % incorrect)
print("Accuracy: %2f" % accuracy)