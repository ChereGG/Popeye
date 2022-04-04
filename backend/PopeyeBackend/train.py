import tensorflow as tf
from decouple import config
from generator import DataGenerator
from models.supervised import vit


def get_loaded_model():
    model = tf.keras.models.load_model('models/conv_model')
    return model


def get_num_of_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return len(lines)

if __name__=="__main__":
    BATCH_SIZE = int(config("BATCH_SIZE"))
    trainGen = DataGenerator("theData", BATCH_SIZE, (8, 8, 13), (1,))
    valGen = DataGenerator("theData", BATCH_SIZE, (8, 8, 13), (1,), train=False)
    # model=conv_classifier.get_conv_model_classification()
    model= vit.create_vit_classifier()
    EPOCHS = int(config("EPOCHS"))
    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='./logsConvModel'),
    ]
    model.fit(trainGen,validation_data=valGen,epochs=EPOCHS, callbacks=callbacks)
