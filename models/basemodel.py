from abc import ABC, abstractmethod
import tensorflow as tf

class BaseModel(ABC, tf.keras.Model):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    @abstractmethod
    def call(self, inputs):
        pass

    @abstractmethod
    def build(self, input_shape):
        pass

    def get_config(self):
        return {"config": self.config}

    @classmethod
    def from_config(cls, config):
        return cls(config['config'])
