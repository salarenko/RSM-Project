import time
import datetime
import numpy as np
import matplotlib as mpl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def run_neural_network(dataFrame, mode):
    print('Start - Neural Network {0}'.format(mode))

    file = open(datetime.datetime.now().strftime("%I-%M%p_%d_%m") + mode + '_decision_tree.txt', 'w')

    X = dataFrame.iloc[:, dataFrame.columns != 'diagnosis'].values
    y = dataFrame.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    file.write('\n\n------------------- Neural Network Algorithm -------------------')
    error = []
    executionTime = []

    for i in range(1, 40):
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
        mlp.fit(X_train, y_train)

        timeStart = time.clock()
        y_pred = mlp.predict(X_test)
        totalTime = time.clock() - timeStart
        error.append(np.mean(y_pred != y_test))
        executionTime.append(totalTime)

        file.write("\nNeural network- Try: {0}".format(i))
        file.write("\nTrainig data: {0} elements".format(len(X_train)))
        file.write("\nTest data: {0} elements".format(len(X_test)))
        file.write("\nExecution time: {0}[s]".format(totalTime))
        file.write("\n-------------------\n")

    file.write("\n {0}".format(classification_report(y_test, y_pred)))
    file.write("\n-------------------\n")

    file.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), executionTime, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Neural Network execution time - mode: {0}'.format(mode))
    plt.xlabel('Number of try')
    plt.ylabel('Execution time [s]')
    plt.show()

    print('Finish - Neural Network {0}'.format(mode))
