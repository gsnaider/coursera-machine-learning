import numpy as np

def cost(X, y, theta):
	m = y.shape[0]
	return 1/(2*m) * np.sum((X.dot(theta) - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
	m = y.shape[0]
	
	for i in range(iterations):
		theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)
	
	return theta
 
