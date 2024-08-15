import numpy as np


# This function counts the occurrences of each unique value in the input array y and prints them.
# Parameters:
# y: np.ndarray, Input array of target values
# Returns: None

def count_occurances(y, label_names = None):
    
    arr = np.unique(y, return_counts=True)
    
    if label_names == None:
        label_names = arr[0]

    for i in range(len(arr[0])):
        print(label_names[i], " occurred ", arr[1][i], " times")

# This function prints the shape of the feature and target arrays
# X: a 2D array of features
# y: a 1D array of targets

def print_shape(X, y):
    print("X shape is ", X.shape)
    print("y shape is ", y.shape)