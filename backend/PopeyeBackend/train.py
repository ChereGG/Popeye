import tensorflow as tf
from decouple import config
from tqdm import tqdm
from generator import DataGenerator
from preprocessing import fen_preprocessing as fp


def get_conv_model_regression():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Input(shape=(8, 8, 6)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='same', kernel_initializer='lecun_normal',
                               activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001),
                              kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(64, activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001),
                              kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=tf.keras.regularizers.L2(0.001),
                              kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def get_conv_model_classification():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 12)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=7, padding='same', kernel_initializer='he_uniform',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


def get_model_dense_classification():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 12)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


def get_loaded_model():
    model = tf.keras.models.load_model('models/conv_model')
    return model


def get_num_of_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return len(lines)

if __name__=="__main__":
    BATCH_SIZE = int(config("BATCH_SIZE"))
    trainGen = DataGenerator("theData", BATCH_SIZE, (8, 8, 12), (1,))
    valGen = DataGenerator("theData", BATCH_SIZE, (8, 8, 12), (1,), train=False)
    model=get_conv_model_classification()
    EPOCHS = int(config("EPOCHS"))
    model.fit(trainGen,validation_data=valGen,epochs=EPOCHS)
