import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def display_data(X):
	print(X.shape)
	print("TODO")



# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('../ex4/ex4data1.mat')
X = data['X']
y = data['y']

m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]

display_data(X[sel, :])

input('Program paused. Press enter to continue.')