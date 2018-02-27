import numpy as np

def cost(X, y, theta):
	m = y.shape[0]
	return 1/(2*m) * np.sum((X.dot(theta) - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
	theta = np.zeros(2)
	return theta

 
