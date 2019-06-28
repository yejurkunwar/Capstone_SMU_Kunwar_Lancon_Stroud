#encoding=utf-8

from models import RNNs
import util
import eval
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd


def RNN_forecasting(dataset, lookBack, lr, inputDim=1, hiddenNum=64, outputDim=1, unit="GRU", epoch=20,
                    batchSize=30, varFlag=False, minLen=15, maxLen=30, step=5):

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,并整理成RNN的输入形式
    train, test = util.divideTrainTest(dataset)

    trainX = None
    trainY = None
    vtrainX = None
    vtrainY = None
    testX = None
    testY = None
    vtestX = None
    vtestY = None

    RNNModel = RNNs.RNNsModel(inputDim, hiddenNum, outputDim, unit, lr)
    if varFlag:
        vtrainX, vtrainY = util.createVariableDataset(train, minLen, maxLen, step)
        vtestX, vtestY = util.createVariableDataset(test, minLen, maxLen, step)
        print("trainX shape is", vtrainX.shape)
        print("trainY shape is", vtrainY.shape)
        print("testX shape is", vtestX.shape)
        print("testY shape is", vtestY.shape)
        RNNModel.train(vtrainX, vtrainY, epoch, batchSize)
    else:
        trainX, trainY = util.createSamples(train, lookBack)
        testX, testY = util.createSamples(test, lookBack)
        print("trainX shape is", trainX.shape)
        print("trainY shape is", trainY.shape)
        print("testX shape is", testX.shape)
        print("testY shape is", testY.shape)
        RNNModel.train(trainX, trainY, epoch, batchSize)

    if varFlag:
        trainPred = RNNModel.predictVarLen(vtrainX, minLen, maxLen, step)
        testPred = RNNModel.predictVarLen(vtestX, minLen, maxLen, step)
        trainPred= trainPred.reshape(-1, 1)
    else:
        trainPred = RNNModel.predict(trainX)
        testPred = RNNModel.predict(testX)
        trainPred = trainPred.reshape(-1, 1)

    if varFlag:
        # 转化一下test的label
        testY = util.transform_groundTruth(vtestY, minLen, maxLen, step)
        testY = testY.reshape(-1, 1)
        testPred = testPred.reshape(-1, 1)
        print("testY", testY.shape)
        print("testPred", testPred.shape)


    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)


    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)


    # util.plot(trainPred,trainY,testPred,testY)

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24
    batch_size = 32
    epoch = 20
    hidden_dim = 64
    lr = 1e-4
    unit = "LSTM"

    ts, data = util.load_data("short_test.csv", columnName="Price")

    trainPred, testPred, mae, mrse, smape = RNN_forecasting(data, lookBack=lag, epoch=epoch, batchSize=batch_size,
                                            varFlag=False, minLen=24, maxLen=48, step=8, unit=unit, lr=lr)

    
