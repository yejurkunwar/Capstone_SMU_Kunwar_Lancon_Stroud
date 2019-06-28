
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# RMSE
def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# MAPE
def calcMAPE(true, pred, epsion = 0.0000000):

    true += epsion
    return np.sum(np.abs((true-pred)/true))/len(true)*100


# SMAPE
def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.sum(np.abs((true-pred)/delim))/len(true)*100
