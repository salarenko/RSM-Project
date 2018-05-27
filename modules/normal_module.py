from modules.components.classifiers.decision_tree import run_decision_tree
from modules.components.classifiers.naive_bayes import run_naive_bayes
from modules.components.classifiers.nearest_neighbours import run_nearest_neighbours
from modules.components.classifiers.neural_network import run_neural_network
from modules.components.clear_data import clear_data
from modules.components.classifiers.svm import run_svm
from setup import run_normal_nn, run_normal_bayes, run_normal_svm, run_normal_decision_tree, run_normal_neural_network


def run_normal_module(loadedData):
    data = clear_data(loadedData)

    if run_normal_nn:
        run_nearest_neighbours(data, 'normal')
    if run_normal_bayes:
        run_naive_bayes(data, 'normal')
    if run_normal_svm:
        run_svm(data, 'normal')
    if run_normal_decision_tree:
        run_decision_tree(data, 'normal')
    if run_normal_neural_network:
        run_neural_network(data, 'normal')