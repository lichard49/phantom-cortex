import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

import subprocess
import matplotlib.pyplot as plt


# plots and saves confusion matrix of model
# params: model to test, test_X (data), test_Y (labels)
# output: visualizes and saves confusion matrix
def plot_confusion_matrix(model, test_X, test_Y, title="Confusion Matrix", filename="default"):
    matrix = sklearn.metrics.plot_confusion_matrix(model, test_X, test_Y)
    plt.title(title)
    plt.show(matrix)
    if filename != "default":
        plt.savefig(filename)

# displays decision tree of RandomForestClassifier model
# params: rfc_model is the model to display, filename is optional for .png file
# output: saves tree visualization as .png, displays through matplotlib
def show_tree(rfc_model, filename='model.png'):
    export_graphviz(rfc_model, out_file='tree.dot', rounded = True, proportion = False, precision = 2, filled = True)
    subprocess.call(['dot', '-Tpng', 'tree.dot', '-o', filename, '-Gdpi=600'])
    # show visualization
    plt.show(filename)

# tests model using given testing dataset
# params: model, test_X (data), test_Y (labels)
def test_model(model, test_X, test_Y):
    return model.score(test_X, test_Y)