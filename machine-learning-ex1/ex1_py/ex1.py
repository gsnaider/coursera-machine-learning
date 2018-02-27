import numpy as np
import matplotlib.pyplot as plt
from warm_up import warm_up_exercise
from linreg import cost

# ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ...')
print('5x5 Identity Matrix:');
warm_up_exercise()

input('Program paused. Press enter to continue.');

# ======================= Pformart 2: Plotting =======================

data = np.loadtxt('../ex1/ex1data1.txt', delimiter=',')
X = data[:,0]
y = data[:,1]
m = y.shape[0]
plt.plot(X,y,'rx')
plt.show()

input('Program paused. Press enter to continue.');

# =================== Pformart 3: Cost and Gradient descent ===================

# Append column of ones to X
ones = np.ones((2, m))
ones[1,:] = X
X = ones.T

# Initialize fitting parameters
theta = np.zeros(2)

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('Testing the cost function ...')
# compute and display initial cost
J = cost(X, y, theta);
print('With theta = [0 ; 0]\nCost computed = %f' % J);
print('Expected cost value (approx) 32.07');

# further testing of the cost function
J = cost(X, y, np.array([-1 , 2]));
print()
print('With theta = [-1 ; 2]\nCost computed = %f\n' % J);
print('Expected cost value (approx) 54.24');

print('Program paused. Press enter to continue.');



# ============= Part 4: Visualizing J(theta_0, theta_1) =============

