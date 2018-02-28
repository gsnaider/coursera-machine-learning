import numpy as np
import matplotlib.pyplot as plt
from logreg import plot_data, cost

# ======================= Part 1: Plotting =======================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.');

data = np.loadtxt('../ex2/ex2data1.txt', delimiter=',')
X = data[:,:2]
y = data[:,2]
plot_data(X, y)
input('\nProgram paused. Press enter to continue.');


# =================== Part 2: Cost and Gradient descent ===================

(m,n) = X.shape

# Append column of ones to X
ones = np.ones((m, n + 1))
ones[:,1:] = X
X = ones


# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
(J, grad) = cost(initial_theta, X, y);

print('Cost at initial theta (zeros): %f' % J);
print('Expected cost (approx): 0.693');
print('Gradient at initial theta (zeros): ');
print(grad);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628');

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2]);
(J, grad) = cost(test_theta, X, y);

print('\nCost at test theta: %f' % J);
print('Expected cost (approx): 0.218');
print('Gradient at test theta: ');
print(grad);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647');

input('\nProgram paused. Press enter to continue.');



