import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from logreg import plot_data, cost, plot_decision_boundary, sigmoid, predict

# ======================= Part 1: Plotting =======================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

data = np.loadtxt('../ex2/ex2data1.txt', delimiter=',')
X = data[:,:2]
y = data[:,2]
plot_data(X, y)
input('\nProgram paused. Press enter to continue.')


# =================== Part 2: Cost and Gradient descent ===================

(m,n) = X.shape

# Append column of ones to X
ones = np.ones((m, n + 1))
ones[:,1:] = X
X = ones


# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
(J, grad) = cost(initial_theta, X, y)

print('Cost at initial theta (zeros): %f' % J)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
(J, grad) = cost(test_theta, X, y)

print('\nCost at test theta: %f' % J)
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

input('\nProgram paused. Press enter to continue.')



## ============= Part 3: Optimizing using minimize =============


res = minimize(cost, initial_theta, (X,y), method='BFGS',jac=True) 
theta = res.x
J = res.fun

# Print theta to screen
print('Cost at theta found by fminunc: %f' % J)
print('Expected cost (approx): 0.203')
print('theta: ')
print(theta)
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')

# Plot Boundary
plot_decision_boundary(theta, X, y)

input('\nProgram paused. Press enter to continue.')

# ============== Part 4: Predict and Accuracies ==============

prob = sigmoid(np.array([1, 45, 85]).dot(theta));
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob);
print('Expected value: 0.775 +/- 0.002\n');

# Compute accuracy on our training set
p = predict(theta, X);

print('Train Accuracy: %f' % (np.mean(p == y) * 100));
print('Expected accuracy (approx): 89.0');