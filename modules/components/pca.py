import pandas as pd

from sklearn.decomposition import PCA


def run_pca(dataFrame, numberOfPrincipals):
    print('Start - PCA')

    principalLabels = []
    classificationColumn = dataFrame.iloc[:, 0].values
    df = dataFrame.drop('diagnosis', 1)

    for i in range(1, numberOfPrincipals + 1):
        principalLabels.append('principal component ' + str(i))

    pca = PCA(n_components=numberOfPrincipals)
    principalComponents = pca.fit_transform(df)

    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=principalLabels)

    principalDf.insert(0, 'diagnosis', classificationColumn)

    print('End - PCA')

    return principalDf
