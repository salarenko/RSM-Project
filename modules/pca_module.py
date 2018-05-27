from modules.components.classifiers.neural_network import run_neural_network
from modules.components.clear_data import clear_data
from modules.components.classifiers.nearest_neighbours import run_nearest_neighbours
from modules.components.classifiers.naive_bayes import run_naive_bayes
from modules.components.classifiers.decision_tree import run_decision_tree
from modules.components.pca import run_pca
from modules.components.classifiers.svm import run_svm
from setup import pca_number_of_dimensions, run_pca_nn, run_pca_bayes, run_pca_svm, run_pca_decision_tree, \
    run_pca_neural_network


def run_pca_module(loadedData):
    data = clear_data(loadedData)
    data = run_pca(data, pca_number_of_dimensions)

    if run_pca_nn:
        run_nearest_neighbours(data, 'pca')
    if run_pca_bayes:
        run_naive_bayes(data, 'pca')
    if run_pca_svm:
        run_svm(data, 'pca')
    if run_pca_decision_tree:
        run_decision_tree(data, 'pca')
    if run_pca_neural_network:
        run_neural_network(data, 'pca')