import tensorflow as tf
import numpy as np


def fen_to_matrix(fen):
    piece_values = {"P": 1, "R": 5, "N": 3, "B": 3, "K": 100, "Q": 9, "p": -1, "r": -5, "n": -3, "b": -3, "k": -100,
                    "q": -9}
    matrix = [[0 for _ in range(8)] for _ in range(8)]
    records = fen.split()
    board = records[0]
    position_rows = board.split("/")
    contor = 0
    for position_row in position_rows:
        for potential_piece in position_row:
            if potential_piece.isalpha():
                matrix[int(contor / 8)][contor % 8] = piece_values[potential_piece] / piece_values["K"]
                contor += 1
            else:
                contor += int(potential_piece)
    return matrix


def fen_to_sparse_matrix6(fen):
    piece_map = {
        "P": [[0 for _ in range(8)] for _ in range(8)],
        "R": [[0 for _ in range(8)] for _ in range(8)],
        "N": [[0 for _ in range(8)] for _ in range(8)],
        "B": [[0 for _ in range(8)] for _ in range(8)],
        "K": [[0 for _ in range(8)] for _ in range(8)],
        "Q": [[0 for _ in range(8)] for _ in range(8)],
    }
    records = fen.split()
    board = records[0]
    position_rows = board.split("/")
    contor = 0
    for position_row in position_rows:
        for potential_piece in position_row:
            if potential_piece.isalpha():
                if potential_piece.isupper():
                    piece_map[potential_piece][int(contor / 8)][contor % 8] = 1
                else:
                    piece_map[potential_piece.upper()][int(contor / 8)][contor % 8] = -1
                contor += 1
            else:
                contor += int(potential_piece)
    return [piece_map["P"], piece_map["R"], piece_map["N"],
            piece_map["B"], piece_map["K"], piece_map["Q"]]


def getTrainDataDenseMatrix(inputFilePath):
    X_train = np.empty(shape=(0, 8, 8))
    Y_train = np.empty(shape=(0))
    with open(inputFilePath, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                continue
            X_train = np.append(X_train, np.array([fen_to_matrix(fen)]), axis=0)
            Y_train = np.append(Y_train, np.array([int(evaluation)]), axis=0)
    return X_train, Y_train


def getTrainDataSparseMatrix6(inputFilePath):
    X_train = np.empty(shape=(0, 6, 8, 8))
    Y_train = np.empty(shape=(0))
    with open(inputFilePath, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                continue
            X_train = np.append(X_train, np.array([fen_to_sparse_matrix6(fen)]), axis=0)
            Y_train = np.append(Y_train, np.array([int(evaluation)]), axis=0)
    return X_train, Y_train


def get_model():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Input(shape=(6, 8, 8)),
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


def get_loaded_model():
    model = tf.keras.models.load_model('models/conv_model')
    return model


def main1():
    model = get_model()
    X_train, Y_train = getTrainDataDenseMatrix("../trainData")
    X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape)
    print(Y_train.shape)
    cb = [tf.keras.callbacks.ModelCheckpoint('conv_model', save_best_only=True),
          tf.keras.callbacks.TensorBoard(log_dir='conv_model_logs')]
    model.fit(X_train, Y_train, validation_split=0.2, epochs=200, callbacks=cb)


def main2():
    model = get_model()
    X_train, Y_train = getTrainDataSparseMatrix6("../trainData")
    print(X_train.shape)
    print(Y_train.shape)
    cb = [tf.keras.callbacks.ModelCheckpoint('conv_model', save_best_only=True),
          tf.keras.callbacks.TensorBoard(log_dir='conv_model_logs')]
    model.fit(X_train, Y_train, validation_split=0.2, epochs=200, callbacks=cb)


if __name__ == '__main__':
    main2()
