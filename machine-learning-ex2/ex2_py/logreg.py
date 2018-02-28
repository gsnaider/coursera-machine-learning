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

	return (0, np.zeros(X.shape[1]))
