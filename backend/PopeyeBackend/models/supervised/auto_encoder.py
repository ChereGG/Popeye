import tensorflow as tf


class AutoEncoderAccuracy(tf.keras.metrics.Metric):

    def __init__(self, thold=0.5, **kwargs):
        super(AutoEncoderAccuracy, self).__init__(**kwargs)
        self.thold = thold
        self.rez = {}

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.thold, dtype=tf.uint8)
        y_true = tf.cast(y_true, dtype=tf.uint8)
        one_true = len(tf.where(y_true == 1))
        zero_true = len(tf.where(y_true == 0))
        correct_ones = len(tf.where(tf.logical_and(y_pred == 1, y_pred == y_true)))
        correct_zeros = len(tf.where(tf.logical_and(y_pred == 0, y_pred == y_true)))
        overall_goods = len(tf.where(y_pred == y_true))
        self.rez["zero_acc"] = tf.squeeze(
            tf.math.divide(tf.convert_to_tensor([correct_zeros], dtype=tf.int32), zero_true))
        self.rez["one_acc"] = tf.squeeze(
            tf.math.divide(tf.convert_to_tensor([correct_ones], dtype=tf.int32), one_true))
        self.rez["overall_acc"] = tf.squeeze(
            tf.math.divide(tf.convert_to_tensor([overall_goods], dtype=tf.int32), tf.size(y_pred)))

    def result(self):
        return self.rez


def get_autoencoder_unet_model():
    inputs = tf.keras.layers.Input(shape=(768,), name="input")
    reshape = tf.keras.layers.Reshape(target_shape=(8, 8, 12), name='reshaper')(inputs)
    # Contraction path
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(reshape)
    # c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    # c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    # c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Expansive path
    u4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = tf.keras.layers.concatenate([u4, c2])
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    # c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c1])
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    # c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    conv_outputs = tf.keras.layers.Conv2D(12, (1, 1), activation="softmax")(c5)
    outputs = tf.keras.layers.Reshape(target_shape=(768,))(conv_outputs)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[AutoEncoderAccuracy()])
    return model


