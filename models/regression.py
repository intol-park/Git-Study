import tensorflow as tf
from .basemodel import BaseModel

class RegressionModel(BaseModel):
    def __init__(self, config):
        super(RegressionModel, self).__init__(config)
        self.hidden_layers = []
        self.output_layer = None
        self.hidden_units = config['hidden_units']

    def build(self, input_shape):
        for units in self.hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')
        super(RegressionModel, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        config = super(RegressionModel, self).get_config()
        config.update({
            'hidden_units': self.hidden_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config)