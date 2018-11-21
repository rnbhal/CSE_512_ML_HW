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

valData = load_data("valData.csv")
valLabels = load_data("valLabels.csv")[:, -1]
trainData = load_data("trainData.csv")
trainingLabel = load_data("trainLabels.csv")[:, -1]
size = valData.shape 
trainData = np.delete(trainData,0, 1) #column 1st has numbering
valData = np.delete(valData,0, 1) #column 1st has numbering 
single = np.full((valData.shape[0],1),1) #Adding nX1 matrix to take bias into account 
valData = np.concatenate( (valData, single),axis=1 )
trainData = np.concatenate( (trainData, single),axis=1 )
loocv = []
validationError = []
trainingError = []
lma = []
RMSA = []
lm = 0.01
while(lm <= 1000.0):
    w,b,obj,cvErrs = ridgeReg(valData, valLabels, lm)
    w_bar = np.concatenate( (w,b ), axis =0 )
    vale = valData.dot(w_bar)-valLabels
    loocv.append(RMSDF(cvErrs, len(cvErrs)))
    validationError.append(RMSDF( vale, vale.shape[0]))
    traininge = trainData.dot(w_bar)-trainingLabel
    trainingError.append(RMSDF( traininge, traininge.shape[0] )  )
    lma.append(lm)
    print ( (lm, RMSDF (cvErrs, len(cvErrs)) ) )
    print("\n")
    lm = lm * 10.0
plt.plot(lma, validationError, label = "ValidationError")
plt.plot(lma, trainingError, label = "TrainingError")
plt.plot(lma, loocv, label = "LOOCVError")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.title("Lambda Vs Root Mean Square Error")
plt.legend()
plt.show()
