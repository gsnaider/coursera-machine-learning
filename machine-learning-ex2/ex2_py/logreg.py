import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, legends=[]):
	pos_x = X[y==1]
	neg_x = X[y==0]
	plt.plot(pos_x[:,0], pos_x[:,1], 'k+')
	plt.plot(neg_x[:,0], neg_x[:,1], 'yo')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	legends.extend(['Admitted','Not admitted'])
	plt.legend(legends)
	plt.show()

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
	m = y.shape[0]
	J = -1 / m * (y.T.dot(np.log(sigmoid(X.dot(theta)))) + (1 -y).T.dot(np.log(1 - sigmoid(X.dot(theta)))))
	grad = 1 / m * X.T.dot(sigmoid(X.dot(theta)) - y)
	return (J, grad)

def plot_decision_boundary(theta, X, y):
	#Two points to define the line
	plot_x1 = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])
	plot_x2 = -1 / theta[2] * (theta[0] + plot_x1 * theta[1])
	plt.plot(plot_x1, plot_x2)
	plot_data(X[:,1:], y, ['Desicion boundary'])

def predict(theta, X):
	return sigmoid(X.dot(theta)) >= 0.5

