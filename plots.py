import matplotlib.pyplot as plt
import numpy as np

# This function creates a 2D scatter plot of the input data.
# Parameters:
# title: str, Title of the plot
# xlabel: str, Label for the x-axis
# ylabel: str, Label for the y-axis
# X: np.ndarray, Input array of features
# y: np.ndarray, Input array of target values
# Returns: None

def scatter2D(title, xlabel, ylabel, X, y, label_names = None):
    unique_classes = np.unique(y)

    if label_names == None:
        label_names = unique_classes
        
    plt.figure(figsize=(16, 8))
    plt.xlabel(xlabel, labelpad=20)
    plt.ylabel(ylabel, labelpad=20)
    
    for i, class_label in enumerate(unique_classes):
      plt.scatter(X[y == class_label, 0], X[y == class_label, 1], label=label_names[i])

    plt.title(title)
    plt.legend()

# This function creates a 3D scatter plot of the input data.
# Parameters:
# title: str, Title of the plot
# xlabel: str, Label for the x-axis
# ylabel: str, Label for the y-axis
# zlabel: str, Label for the z-axis
# X: np.ndarray, Input array of features
# y: np.ndarray, Input array of target values
# Returns: None

def scatter3D(title, xlabel, ylabel, zlabel, X, y, label_names = None):
    unique_classes = np.unique(y)

    if label_names == None:
        label_names = unique_classes

    # Create a new figure
    fig = plt.figure(figsize=(32, 16))
    # Add a subplot with 3D prjection
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    for i, class_label in enumerate(unique_classes):
      ax.scatter(X[y == class_label, 0], X[y == class_label, 1], X[y == class_label, 2], label=label_names[i])
    
    # Set labels
    ax.set_xlabel(xlabel, labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_zlabel(zlabel, labelpad=20)

    ax.set_box_aspect(None, zoom=0.75)

    # Set title
    plt.title(title)

    plt.legend()
    # Show the plot
    plt.show()