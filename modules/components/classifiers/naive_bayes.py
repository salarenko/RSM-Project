import time
import datetime
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

def run_naive_bayes(dataFrame, mode):
    print('Start - Naive Bayes {0}'.format(mode))
    file = open(datetime.datetime.now().strftime("%I-%M%p_%d_%m") + mode + '_naive_bayes.txt', 'w')
    model = GaussianNB()

    X = dataFrame.iloc[:, dataFrame.columns != 'diagnosis'].values
    y = dataFrame.iloc[:, 0].values

    file.write('\n\n----------------- Bayes Algorithm ------------------')
    error = []
    executionTime = []

    for i in range(1, 40):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        timeStart = time.clock()
        y_pred = model.predict(X_test)
        totalTime = time.clock() - timeStart

        error.append(np.mean(y_pred != y))
        executionTime.append(totalTime)

        file.write("\nNaive Bayes - try: {0}".format(i))
        file.write("\nTrainig data: {0} elements".format(len(X_train)))
        file.write("\nTest data: {0} elements".format(len(X_test)))
        file.write("\nExecution time: {0}[s]".format(totalTime))
        file.write("\n-------------------\n")
    file.write("\n {0}".format(classification_report(y_test, y_pred)))
    file.write("\n-------------------\n")

    file.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), executionTime, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=1)
    plt.title('Naive Bayes classifier execution time - mode: {0}'.format(mode))
    plt.xlabel('Number of try')
    plt.ylabel('Execution time [s]')
    plt.show()

    print('Finish - Naive Bayes {0}'.format(mode))
