import numpy as np
import matplotlib.pyplot as plt
from warm_up import warm_up_exercise


# ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ...')
print('5x5 Identity Matrix:');
warm_up_exercise()

input('Program paused. Press enter to continue.');

# ======================= Part 2: Plotting =======================

data = np.loadtxt('../ex1/ex1data1.txt', delimiter=',')
X = data[:,0]
y = data[:,1]
plt.plot(X,y,'rx')
plt.show()

# =================== Part 3: Cost and Gradient descent ===================




# ============= Part 4: Visualizing J(theta_0, theta_1) =============

