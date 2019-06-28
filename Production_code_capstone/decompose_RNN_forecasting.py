#encoding=utf-8

import util
from models import decompose
import eval
from naive_RNN_forecasting import RNN_forecasting
import time
import matplotlib.pyplot as plt


def decompose_RNN_forecasting(ts, dataset, freq, lag, epoch=20, hidden_num=64,
                              batch_size=32, lr=1e-3, unit="GRU", varFlag=False, maxLen=48, minLen=24, step=8):


    trend, seasonal, residual = decompose.ts_decompose(ts, freq)
    print("trend shape:", trend.shape)
    print("peroid shape:", seasonal.shape)
    print("residual shape:", residual.shape)


    resWin = trendWin = lag
    t1 = time.time()
    trTrain, trTest, MAE1, MRSE1, SMAPE1 = RNN_forecasting(trend, lookBack=lag, epoch=epoch, batchSize=batch_size, hiddenNum=hidden_num,
                                            varFlag=varFlag, minLen=minLen, maxLen=maxLen, step=step, unit=unit, lr=lr)
    resTrain, resTest, MAE2, MRSE2, SMAPE2 = RNN_forecasting(residual, lookBack=lag, epoch=epoch, batchSize=batch_size, hiddenNum=hidden_num,
                                            varFlag=varFlag, minLen=minLen, maxLen=maxLen, step=step, unit=unit, lr=lr)
    t2 = time.time()
    print(t2-t1)

    print("trTrain shape:", trTrain.shape)
    print("resTrain shape:", resTrain.shape)


    trendPred, resPred = util.align(trTrain, trTest, trendWin, resTrain, resTest, resWin)
    print("trendPred shape is", trendPred.shape)
    print("resPred shape is", resPred.shape)


    # finalPred = trendPred+seasonal+resPred

    trainPred = trTrain+seasonal[trendWin:trendWin+trTrain.shape[0]]+resTrain
    testPred = trTest+seasonal[2*resWin+resTrain.shape[0]:]+resTest


    data = dataset[freq//2:-(freq//2)]
    trainY = data[trendWin:trendWin+trTrain.shape[0]]
    testY = data[2*resWin+resTrain.shape[0]:]

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    plt.plot(testY, label='ground-truth')
    plt.plot(testPred, label='prediction')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Price", fontsize=10)
    plt.legend()
    foo_fig = plt.gcf()
    foo_fig.savefig('Price_r416xlarge_Linux.eps', format='eps', dpi=1000, bbox_inches='tight')
    plt.show()

    return trainPred, testPred, MAE, MRSE, SMAPE



if __name__ == "__main__":

    lag = 24  # if using varFlag, lag == maxLen
    batch_size = 32
    epoch = 12
    hidden_dim = 64
    unit = "LSTM"
    lr = 1e-4
    freq = 8
    varFlag = True
    minLen = 12
    maxLen = 24
    step = 6

    ts, data = util.load_data("short_test.csv", columnName="Price")

    trainPred, testPred, mae, mrse, smape = decompose_RNN_forecasting(ts, data, lag=lag, freq=freq, unit=unit,
                                                                     varFlag=varFlag, minLen=minLen, maxLen=maxLen,
                                                                     step=step, epoch=epoch, hidden_num=hidden_dim,
                                                                     lr=lr, batch_size=batch_size)
