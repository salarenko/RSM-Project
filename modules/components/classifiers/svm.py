import time
import datetime
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm


def run_svm(dataFrame, mode):
    print('Start - SVM {0}'.format(mode))

    file = open(datetime.datetime.now().strftime("%I-%M%p_%d_%m") + mode + '_svm.txt', 'w')

    X = dataFrame.iloc[:, dataFrame.columns != 'diagnosis'].values
    y = dataFrame.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

    file.write('\n\n------------------- SVM Algorithm -------------------')
    error = []
    executionTime = []

    for i in range(1, 40):
        model = svm.SVC(kernel='linear', C=1, gamma=1)
        model.fit(X_train, y_train)
        model.score(X_train, y_train)

        timeStart = time.clock()
        y_pred = model.predict(X_test)
        totalTime = time.clock() - timeStart
        error.append(np.mean(y_pred != y_test))
        executionTime.append(totalTime)

        file.write("\nSVM - Try: {0}".format(i))
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
    plt.title('SVM execution time - mode: {0}'.format(mode))
    plt.xlabel('Number of try')
    plt.ylabel('Execution time [s]')
    plt.show()

    print('Finish - SVM {0}'.format(mode))
