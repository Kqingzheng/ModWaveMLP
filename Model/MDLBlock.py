import tensorflow as tf
from Parameters import Parameters

class FC_DE(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, **kw):
        super(FC_DE, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = []
        for i in range(hyperparams.block_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(hyperparams.hidden_units,
                                      activation=tf.nn.tanh,
                                      kernel_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay),
                                      name=f"fc_{i}")
            )
        self.forecast = tf.keras.layers.Dense(self.output_size, activation=None, name="forecast")
        self.backcast = tf.keras.layers.Dense(self.input_size, activation=None, name="backcast")

    def call(self, inputs, training=False):
        h = self.fc_layers[0](inputs)
        for i in range(1, self.hyperparams.block_layers):
            h = self.fc_layers[i](h)
        backcast_out = self.backcast(h)
        backcast = tf.keras.activations.relu(inputs - backcast_out)
        forecast_out = self.forecast(h)

        return backcast, forecast_out



class FC_DEAdd(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, **kw):
        super(FC_DEAdd, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = []
        for i in range(hyperparams.block_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(hyperparams.hidden_units,
                                      activation=tf.nn.tanh,
                                      kernel_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay),
                                      name=f"fc_{i}")
            )
        self.forecast = tf.keras.layers.Dense(self.output_size, activation=None, name="forecast")
        self.backcast = tf.keras.layers.Dense(self.input_size, activation=None, name="backcast")

    def call(self, inputs, training=False):
        h = self.fc_layers[0](inputs)
        for i in range(1, self.hyperparams.block_layers):
            h = self.fc_layers[i](h)
        backcast_out = self.backcast(h)
        backcast = tf.keras.activations.relu(inputs + backcast_out)
        forecast_out = self.forecast(h)

        return backcast, forecast_out

class MDLBlock(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, **kw):
        super(MDLBlock, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.blocks = []
        for i in range(self.hyperparams.blocks):
            self.blocks.append(FC_DE(hyperparams=hyperparams,
                                       input_size=self.input_size,
                                       output_size=hyperparams.horizon,
                                       name=f"block_{i}"))
    def call(self,history):
        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.hyperparams.blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:, :, :self.hyperparams.horizon]
        return backcast, forecast_out
       