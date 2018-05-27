import time
import datetime
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def run_nearest_neighbours(dataFrame, mode):
    print('Start - Nearest Neighbours {0}'.format(mode))

    file = open(datetime.datetime.now().strftime("%I-%M%p_%d_%m") + mode + '_nearest_neighbours.txt', 'w')

    X = dataFrame.iloc[:, dataFrame.columns != 'diagnosis'].values
    y = dataFrame.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    file.write('\n\n------------------- NN Algorithm -------------------')
    error = []
    executionTime = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        timeStart = time.clock()
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        totalTime = time.clock() - timeStart
        error.append(np.mean(pred_i != y_test))
        executionTime.append(totalTime)

        file.write("\nNeighbours: {0}".format(i))
        file.write("\nTrainig data: {0} elements".format(len(X_train)))
        file.write("\nTest data: {0} elements".format(len(X_test)))
        file.write("\nExecution time: {0}[s]".format(totalTime))
        file.write("\n {0}".format(classification_report(y_test, pred_i)))
        file.write("\n-------------------\n")

    file.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value - mode: {0}'.format(mode))
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), executionTime, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Nearest neighbours classifier execution time - mode: {0}'.format(mode))
    plt.xlabel('Number of neighbours')
    plt.ylabel('Execution time [s]')
    plt.show()

    print('Finish - Nearest Neighbours {0}'.format(mode))
