import tensorflow as tf

def get_conv_model_classification():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 13)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'], run_eagerly=True)
    return model
