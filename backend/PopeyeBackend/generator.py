import tensorflow as tf
import numpy as np
from preprocessing import fen_preprocessing as fp


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, fileName, batchSize, inputSize, outputSize, validationSplit=0.2, train=True,
                 shuffle=True):
        self.__fileName = fileName
        self.__batchSize = batchSize
        self.__shuffle = shuffle
        self.__inputSize = inputSize
        self.__outputSize = outputSize
        self.__train = train
        self.__validationSplit = validationSplit
        with open(self.__fileName, 'r') as file:
            numOfAllData = self.__getNumOfData()
            if self.__train:
                size = int(numOfAllData * (1 - self.__validationSplit))
                self.__lines = file.readlines()[:size]
            else:
                size = int(numOfAllData * (1 - self.__validationSplit))
                self.__lines = file.readlines()[size:]
        self.__numOfData = len(self.__lines)
        self.on_epoch_end()

    def __getNumOfData(self):
        with open(self.__fileName, 'r') as file:
            rez = len(file.readlines())
        return rez

    def __len__(self):
        return int(np.ceil(self.__numOfData / self.__batchSize))

    def __getitem__(self, index):
        if index == self.__len__() - 1 and self.__numOfData % self.__batchSize != 0:
            X = np.zeros(shape=(self.__numOfData % self.__batchSize, *self.__inputSize))
            Y = np.zeros(shape=(self.__numOfData % self.__batchSize, *self.__outputSize))
            for i in range(self.__numOfData % self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix12(fen))
                if int(eval) < -100:
                    Y[i] = np.array([0])
                elif -100 < int(eval) < 100:
                    Y[i] = np.array([1])
                else:
                    Y[i] = np.array([2])
        else:
            X = np.zeros(shape=(self.__batchSize, *self.__inputSize))
            Y = np.zeros(shape=(self.__batchSize, *self.__outputSize))
            for i in range(self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix12(fen))
                if int(eval) < -100:
                    Y[i] = np.array([0])
                elif -100 < int(eval) < 100:
                    Y[i] = np.array([1])
                else:
                    Y[i] = np.array([2])
        return X, Y

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__lines)
