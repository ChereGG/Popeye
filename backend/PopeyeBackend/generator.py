import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from preprocessing import fen_preprocessing as fp
from models.supervised.auto_encoder import AutoEncoderAccuracy



class DataGeneratorEncoder(tf.keras.utils.Sequence):

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
                self.__lines = file.readlines()[size]
        self.__numOfData = len(self.__lines)
        self.on_epoch_end()

    def __getNumOfData(self):
        with open(self.__fileName, 'r') as file:
            rez = len(file.readlines())
        return rez

    def __len__(self):
        return int(np.ceil(self.__numOfData / self.__batchSize))

    def __getitem__(self, index):
        if index == self.__len__() - 1 and self.__numOfData % self.__batchSize > 0:
            X = np.zeros(shape=(self.__numOfData % self.__batchSize, *self.__inputSize))
            for i in range(self.__numOfData % self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
            X, Y = np.reshape(X, newshape=(self.__numOfData % self.__batchSize, 832)), np.reshape(X, newshape=(
                self.__numOfData % self.__batchSize, 832))
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(X), y=X.flatten())
            class_weights = dict(enumerate(class_weights))
            sample_weights = class_weight.compute_sample_weight(class_weights, X.flatten())
            sample_weights = np.reshape(sample_weights, newshape=(self.__numOfData % self.__batchSize, 832))
        else:
            X = np.zeros(shape=(self.__batchSize, *self.__inputSize))
            for i in range(self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
            X, Y = np.reshape(X, newshape=(self.__batchSize, 832)), np.reshape(X, newshape=(self.__batchSize, 832))
            sample_weights = class_weight.compute_sample_weight('balanced', X.flatten())
            sample_weights = np.reshape(sample_weights, newshape=(self.__batchSize, 832))

        return X, Y, np.expand_dims(sample_weights, axis=-1)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__lines)


class DataGeneratorEvals(tf.keras.utils.Sequence):

    def __init__(self, fileName, batchSize, inputSize, outputSize, validationSplit=0.2, train=True,
                 shuffle=True):
        self.__fileName = fileName
        self.__batchSize = batchSize
        self.__shuffle = shuffle
        self.__inputSize = inputSize
        self.__outputSize = outputSize
        self.__train = train
        self.__validationSplit = validationSplit
        self.__encoder = get_encoder()
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
        if index == self.__len__() - 1 and self.__numOfData % self.__batchSize > 0:
            X = np.zeros(shape=(self.__numOfData % self.__batchSize, *self.__inputSize))
            Y = np.zeros(shape=(self.__numOfData % self.__batchSize, *self.__outputSize))
            for i in range(self.__numOfData % self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
                if int(eval) < -100:
                    Y[i] = np.array([0])
                elif -100 < int(eval) < 100:
                    Y[i] = np.array([1])
                else:
                    Y[i] = np.array([2])
            sample_weights = class_weight.compute_sample_weight('balanced', Y.flatten())
            sample_weights = np.expand_dims(sample_weights, axis=-1)
            X = np.reshape(X, newshape=(self.__numOfData % self.__batchSize, 768))

        else:
            X = np.zeros(shape=(self.__batchSize, *self.__inputSize))
            Y = np.zeros(shape=(self.__batchSize, *self.__outputSize))
            for i in range(self.__batchSize):
                fen = self.__lines[index * self.__batchSize + i].strip().split(":")[0]
                eval = self.__lines[index * self.__batchSize + i].strip().split(":")[1]
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
                if int(eval) < -100:
                    Y[i] = np.array([0])
                elif -100 < int(eval) < 100:
                    Y[i] = np.array([1])
                else:
                    Y[i] = np.array([2])
            sample_weights = class_weight.compute_sample_weight('balanced', Y.flatten())
            sample_weights = np.expand_dims(sample_weights, axis=-1)
            X = np.reshape(X, newshape=(self.__batchSize, 768))

        X = self.__encoder(X)
        return X, Y, sample_weights

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__lines)


def get_loaded_auto_encoder_model():
    auto_enc = tf.keras.models.load_model('models/auto_encoder_unet_model_8',
                                          custom_objects={"AutoEncoderAccuracy": AutoEncoderAccuracy()})
    return auto_enc


def get_encoder():
    auto_enc = get_loaded_auto_encoder_model()
    encoder = auto_enc.layers[:10]
    model = tf.keras.models.Sequential(encoder)
    return model


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
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
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
                X[i] = np.transpose(fp.fen_to_sparse_matrix13(fen))
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
