import tensorflow as tf
import numpy as np
from preprocessing import fen_preprocessing as fp


def getTrainData(inputFilePath):
    X_train = np.empty(shape=(0, 8, 8))
    Y_train = np.empty(shape=(0))
    with open(inputFilePath, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                # if "0" in evaluation:
                #     evaluation = 1000000000
                # else:
                #     evaluation = (1 / int(evaluation.split("#")[1])) * 10000
                continue
            X_train = np.append(X_train, np.array([fp.fen_to_matrix(fen)]), axis=0)
            Y_train = np.append(Y_train, np.array([int(evaluation)]), axis=0)
    return X_train, Y_train


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
