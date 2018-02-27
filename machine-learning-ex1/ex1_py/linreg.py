import numpy as np

def cost(X, y, theta):
	m = y.shape[0]
	return 1/(2*m) * np.sum((X.dot(theta) - y) ** 2) 
