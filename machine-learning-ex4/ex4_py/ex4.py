import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy.io import loadmat
from scipy.optimize import minimize
from math import *


def append_ones_row(a):
    (rows, cols) = a.shape
    ones = np.ones((rows + 1, cols))
    ones[1:, :] = a
    return ones


def append_ones_col(a):
    (rows, cols) = a.shape
    ones = np.ones((rows, cols + 1))
    ones[:, 1:] = a
    return ones


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def display_data(X):
    example_width = int(round(sqrt(X.shape[1])))

    # Compute rows, cols
    (m, n) = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(floor(sqrt(m)))
    display_cols = int(ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch

            # Get the max value of the pach
            max_val = np.max(np.abs(X[curr_ex, :]))

            height_idx = pad + j * (example_height + pad)
            width_idx = pad + i * (example_width + pad)
            display_array[height_idx: height_idx + example_height, width_idx: width_idx + example_width] = np.reshape(
                X[curr_ex, :],
                (example_height,
                 example_width)) / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break
    plt.imshow(display_array.T, cmap='gray')
    plt.show()


def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 0.0001
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        (loss1, grad) = J(theta - perturb)
        (loss2, grad) = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad


def check_nn_gradients(reg_param=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    nn_params = np.append(theta1.ravel(), theta2.ravel())

    # Reusing debug_initialize_weights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(m), num_labels)
    cost_fun = functools.partial(nn_cost_function, input_layer_size=input_layer_size,
                                 hidden_layer_size=hidden_layer_size, num_labels=num_labels, X=X, y=y,
                                 reg_param=reg_param)

    (cost, grad) = cost_fun(nn_params)
    numgrad = compute_numerical_gradient(cost_fun, nn_params)

    print(np.column_stack((numgrad, grad)))
    print(
        'The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n' +
          'the relative difference will be small (less than 1e-9). \n' +
          '\nRelative Difference: %g\n' % diff)


def debug_initialize_weights(l_out, l_in):
    w = np.zeros((l_out, l_in + 1))
    return np.reshape(np.sin(np.arange(w.size)), w.shape) / 10


def rand_initialize_weights(l_in, l_out):
    epsilon = 0.12
    return np.random.random((l_out, 1 + l_in)) * 2 * epsilon - epsilon


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param):
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (hidden_layer_size + 1)))

    (m, n) = X.shape

    # Append column of ones to X
    X = append_ones_col(X)

    # Map y to binary vectors
    y_bin = np.zeros((m, num_labels))
    y_bin[[np.arange(m), (y - 1).ravel()]] = 1
    y = y_bin

    a1 = X.T
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = append_ones_row(a2)
    z3 = theta2.dot(a2)
    h = sigmoid(z3).T

    J = -1 / m * np.sum((y * np.log(h)) + (1 - y) * np.log(1 - h)) + reg_param / (2 * m) * (
            np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

    delta_output = h - y
    delta_hidden = delta_output.dot(theta2[:, 1:]) * sigmoid_gradient(z2).T

    theta1_grad = 1 / m * a1.dot(delta_hidden).T
    theta2_grad = 1 / m * a2.dot(delta_output).T

    grad = np.append(theta1_grad.ravel(), theta2_grad.ravel())

    return (J, grad)


def predict(theta1, theta2, X):
    (m, n) = X.shape

    # Append column of ones to X
    X = append_ones_col(X)

    h1 = sigmoid(X.dot(theta1.T))
    h1 = append_ones_col(h1)
    h2 = sigmoid(h1.dot(theta2.T))

    return np.argmax(h2, axis=1) + 1


# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('\nLoading and Visualizing Data ...\n')

data = loadmat('../ex4/ex4data1.mat')
X = data['X']
y = data['y']

m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]

display_data(X[sel, :])

input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
nn_params = loadmat('../ex4/ex4weights.mat')

theta1 = np.array(nn_params['Theta1'])
theta2 = np.array(nn_params['Theta2'])

nn_params = np.append(theta1.ravel(), theta2.ravel())

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
reg_param = 0

(J, grad) = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)' % J)

input('Program paused. Press enter to continue.\n')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
reg_param = 1

(J, grad) = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)\n' % J)

input('Program paused. Press enter to continue.\n')

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(g)
print('\n\n')

input('Program paused. Press enter to continue.\n')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

initial_nn_params = np.append(initial_theta1.ravel(), initial_theta2.ravel())

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.

print('\nChecking Backpropagation... \n');

#  Check gradients by running checkNNGradients
check_nn_gradients()

input('\nProgram paused. Press enter to continue.\n');

## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.

print('\nTraining Neural Network... \n')

#  You should also try different values of lambda
reg_param = 1

# Create "short hand" for the cost function to be minimized
cost_fun = functools.partial(nn_cost_function, input_layer_size=input_layer_size,
                             hidden_layer_size=hidden_layer_size, num_labels=num_labels, X=X, y=y,
                             reg_param=reg_param)
res = minimize(cost_fun, initial_nn_params, method='TNC', jac=True, options={'maxiter': 500})

nn_params = res.x
J = res.fun

# Obtain Theta1 and Theta2 back from nn_params
theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, (hidden_layer_size + 1)))

input('Program paused. Press enter to continue.\n')

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('\nVisualizing Neural Network... \n')

display_data(theta1[:, 1:])

input('\nProgram paused. Press enter to continue.\n');

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(theta1, theta2, X)

print('\nTraining Set Accuracy:')

print(np.mean((pred == y.ravel()) * 1.0) * 100)
