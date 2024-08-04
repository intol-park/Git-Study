import tensorflow as tf

def load_dataset(name):
    if name == 'boston_housing':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
        return (x_train, y_train), (x_test, y_test)
    else:
        raise ValueError("Dataset not found")
