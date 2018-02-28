import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
	pos_x = X[y==1]
	neg_x = X[y==0]
	plt.plot(pos_x[:,0], pos_x[:,1], 'k+')
	plt.plot(neg_x[:,0], neg_x[:,1], 'yo')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(['Admitted','Not admitted'])
	plt.show()

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
	m = y.shape[0]
	J = -1 / m * (y.T.dot(np.log(sigmoid(X.dot(theta)))) + (1 -y).T.dot(np.log(1 - sigmoid(X.dot(theta)))))
	grad = 1 / m * X.T.dot(sigmoid(X.dot(theta)) - y)
	return (J, grad)
