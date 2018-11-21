import scipy, numpy as np
from numpy.linalg import inv
import pandas as pd
import csv
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
import pickle

def load_data(filename):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    data = np.matrix(x).astype('float')
    return data

def RMSDF(RMS, N):
    RMS = np.square(RMS)
    return (np.sqrt(RMS.sum()/N))

def ridgeReg(X, Y, l):
    C = (X.T.dot(X) + np.identity(X.shape[1], dtype=int)*l) # C = Xt * X + l
    IC = inv(C)
    N = X.shape[0]
    D = X.T.dot(Y)
    w_bar = IC.dot(D)
    w = w_bar[:-1]
    b = w_bar[-1]
    obj = l*(w_bar.T.dot(w_bar)) + np.sum((X.dot(w_bar) - Y))
    RMSD = np.zeros((N,1), dtype=float)
    for i in range(0,N):
        xi = X[i,:]
        yi = Y[i,:]
        numerator = (w_bar.T.dot(xi.T)) - yi  # kXn nX1 = kX1
        D1 = IC.dot(xi.T)
        denominator = 1 - xi.dot(D1)
        error = numerator/denominator[0][0]
        RMSD[i] = error[0][0]
        # if (i+1)%500 == 0:
        #   print (str(i+1),RMSDF(RMSD, i) )
    return w, b, obj, RMSD

def RMSE(X, Y, l, Xt, Yt):
    C = (X.T.dot(X) + np.identity(X.shape[1], dtype=int)*l) # C = Xt * X + l
    IC = inv(C)
    N = Xt.shape[0]
    D = X.T.dot(Y)
    w_bar = IC.dot(D)
    RMSD = []
    RMSD.append((Xt.dot(w_bar)- Yt))
    return w_bar, RMSDF(RMSD, N)

def solve(X, Y, l):
    C = (X.T.dot(X) + np.identity(X.shape[1], dtype=int)*l) # C = Xt * X + l
    IC = inv(C)
    N = X.shape[0]
    D = X.T.dot(Y)
    RMSD = []
    theta = IC.dot(D)
    RMSD.append(X.dot(theta)- Y)
    return theta, RMSDF(RMSD, RMSD[0].shape[0])


ltrainData = load_data("trainData.csv")
loutputData = load_data("trainLabels.csv")[:, -1]
trainData = load_data("valData.csv")
outputData = load_data("valLabels.csv")[:, -1]
size = ltrainData.shape 
ltrainData = np.delete(ltrainData,0, 1) #column 1st has numbering 
trainData = np.delete(trainData,0, 1) #column 1st has numbering 


ltrainData = np.concatenate( (ltrainData, trainData),axis=0 )
single = np.full((ltrainData.shape[0],1),1) #Adding nX1 matrix to take bias into account
ltrainData = np.concatenate( (ltrainData, single),axis=1 )
loutputData = np.concatenate( (loutputData, outputData),axis=0 )
lm1 = 0.775824
f_theta, f_RMSE = solve(ltrainData, loutputData, lm1)
testData = load_data("testData.csv")
size = testData.shape 
testData = np.delete(testData,0, 1) #column 1st has numbering 
single = np.full((testData.shape[0],1),1) #Adding nX1 matrix to take bias into account 
testData = np.concatenate( (testData, single),axis=1 )
test_y = testData.dot(f_theta)
test_y = np.concatenate((((np.matrix(np.arange(5000))).T,test_y)),axis=1)
np.savetxt('predTestLabels.csv',test_y,fmt='%.10f', delimiter=',')


print(lm1, f_RMSE)