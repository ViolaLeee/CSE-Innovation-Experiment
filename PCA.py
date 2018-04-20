import numpy as np


def loadCSVdata(file):
    tmp = np.loadtxt(file, dtype=np.str, delimiter=",")
    data = tmp[1:1514, 2:22].astype(np.float)# load data
    return data


def loadCSVlabel(file):
    tmp = np.loadtxt(file, dtype=np.str, delimiter=",")
    lab = tmp[0, 2:22].astype(np.str)# load labels
    label = []

    # store the labels into an array
    for items in range(20):
        label.append(lab[items])
    return label

    """reader = csv.reader(file)
    data = []
    num = 0
    for item in reader:
        num+=1
        if(num <= 100):
            data.append(item)
            print(item)
    return data"""


def disMean(matrix):
    matrix_discard_mean = np.zeros((matrix.shape), dtype = float)# array to store the result of discard mean values
    [rows, cols] = matrix.shape
    mat_mean = []# array to store the mean values

    # compute mean values
    for j in range(cols):
        mat_mean.append((matrix.sum(axis=0)[j])/rows)# column sum/number of rows

    # discard the mean values
    for i in range(rows):
        for j in range(cols):
            matrix_discard_mean[i][j] = matrix[i][j] - mat_mean[j]
    a = np.mat(matrix_discard_mean)
    return matrix_discard_mean


def covMatrix(matrix):
    transposeMatrix = matrix.T
    covMat = np.cov(transposeMatrix)
    return covMat


def sortEigenValue(list):
    sortedList = sorted(list, reverse=True)
    sortedEigen = np.zeros((20, 2))
    count = 0
    for items in sortedList:
        for i in range(20):
            if(items == list[i]):
                sortedEigen[count][0] = items
                sortedEigen[count][1] = i
        count += 1
    return sortedEigen


def selectEigenVector(sortedEigVal, eigenVec):
    selectEigenVector = np.zeros((5,20))
    for i in range(5):
        selectEigenVector[i] = eigenVec[int(sortedEigVal[i][1])]
    return selectEigenVector


def finalMatrix(data, eigenVecMat):
    handledMatrix = np.dot(data, eigenVecMat)
    return handledMatrix


if __name__ == "__main__":
    stocklabel = loadCSVlabel("300001.SZ.CSV")# array of stock label
    # print(stocklabel)
    stockdata = loadCSVdata("300001.SZ.CSV")# array of stock data
    matrix_dis_mean = disMean(stockdata)# array of discard the mean values of each feature
    # print(disMean(stockdata))
    covarianceMatrix = covMatrix(matrix_dis_mean)# covariance matrix
    # print(covarianceMatrix)
    eigenValue, eigenVector = np.linalg.eig(covarianceMatrix)# eigenValue and eigenVector
    # print(eigenValue)
    # print(eigenVector)
    sortedEigenValue = sortEigenValue(eigenValue)# array of sorted eigenValue
    # print(sortedEigenValue)
    selectedEigenVector = selectEigenVector(sortedEigenValue, eigenVector)# array of selected eigenVector
    # print(selectedEigenVector)
    eigenVectorMatrix = selectedEigenVector.T# transpose the selected eigen vector to compute
    # print(eigenVectorMatrix)
    handledData = finalMatrix(matrix_dis_mean, eigenVectorMatrix)
    # print(handledData)
    np.savetxt("handled300001.CSV", handledData, delimiter=",")# save data to a csv file