#encoding=utf-8

# import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.layers import Dropout
from keras.regularizers import l2
import numpy as np
from keras import optimizers


class RNNsModel(object):

    def __init__(self, inputDim, hiddenNum, outputDim, unit, lr):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
        self.buildModel(unit)


    def buildModel(self, unit="GRU"):

        self.model = Sequential()
        if unit == "GRU":
            self.model.add(GRU(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "LSTM":
            self.model.add(LSTM(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "RNN":
            self.model.add(SimpleRNN(self.hiddenNum, input_shape=(None, self.inputDim)))
        self.model.add(Dense(self.outputDim))
        self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=["mean_absolute_percentage_error"])


    def train(self, trainX, trainY, epoch, batchSize):
        self.model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1, validation_split=0.0)


    def predict(self,testX):
        pred = self.model.predict(testX)
        return pred


    def predictVarLen(self, vtestX, minLen, maxLen, step):
        lagNum = (maxLen-minLen) // step + 1
        predAns = []
        pred = self.model.predict(vtestX)
        for i in range(0, len(pred), lagNum):
            predAns.append(np.mean(pred[i:i+lagNum]))
        return np.array(predAns)
