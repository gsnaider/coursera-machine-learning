import numpy as np
import matplotlib.pyplot as plt

def cost(X, y, theta):
	m = y.shape[0]
	return 1/(2*m) * np.sum((X.dot(theta) - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
	m = y.shape[0]
	costs = []
	for i in range(iterations):
		theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)
		costs.append(cost(X,y,theta))
	
	plt.plot(range(iterations), costs)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()	
	return theta
 
