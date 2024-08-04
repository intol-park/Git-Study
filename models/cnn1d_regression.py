import tensorflow as tf
from .basemodel import BaseModel

class CNN1DRegressionModel(BaseModel):
    def __init__(self, config):
        super(CNN1DRegressionModel, self).__init__(config)
        self.conv_layers = []
        self.flatten = None
        self.dense = None
        self.output_layer = None
        self.config = config

    def build(self, input_shape):
        for conv_layer_config in self.config['conv_layers']:
            self.conv_layers.append(tf.keras.layers.Conv1D(
                filters=conv_layer_config['filters'],
                kernel_size=conv_layer_config['kernel_size'],
                activation=conv_layer_config['activation'],
                input_shape=input_shape
            ))
            self.conv_layers.append(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.config['dense_units'], activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)

    def get_config(self):
        config = super(CNN1DRegressionModel, self).get_config()
        config.update({
            'conv_layers': self.config['conv_layers'],
            'dense_units': self.config['dense_units']
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config)
