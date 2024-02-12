from keras.layers.core import *
import tensorflow as tf
from keras.models import *
from Parameters import Parameters
from Model.MoDWaveLayer import MoDWaveLayer


class MoDWaveMLP:
    def __init__(self, hyperparams: Parameters, name: str = 'modwave', logdir: str = 'logs', num_nodes: int = 100):
        super(MoDWaveMLP, self).__init__()
        self.hyperparams = hyperparams
        self.name = name
        self.logdir = logdir
        self.num_nodes = num_nodes
        self.input_size = self.hyperparams.history_length + self.hyperparams.node_id_dim + self.num_nodes * self.hyperparams.history_length

        self.modwave_layers = []
        for i in range(hyperparams.num_stacks):
            self.modwave_layers.append(MoDWaveLayer(hyperparams=hyperparams,
                                                  input_size=self.input_size,
                                                  output_size=hyperparams.horizon,
                                                  num_nodes=self.num_nodes,
                                                  name=f"modwave_{i}")
                                      )

        inputs, outputs = self.get_model()
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.inputs = inputs
        self.inputs = outputs
        self.model = model

    def get_model(self):
        history_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length), name='history')
        time_of_day_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                               name='time_of_day')
        day_in_week_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                               name='day_in_week')
        node_id_in = tf.keras.layers.Input(shape=(self.num_nodes, 1), dtype=tf.uint16, name='node_id')
        wt1_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                       name='wt1')
        wt2_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                       name='wt2')
        wt3_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                       name='wt3')
        wt4_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length),
                                       name='wt4')
        backcast, forecast = self.modwave_layers[0](history_in=history_in, node_id_in=node_id_in,
                                                   time_of_day_in=time_of_day_in, day_in_week_in=day_in_week_in,
                                                   wt1=wt1_in, wt2=wt2_in, wt3=wt3_in, wt4=wt4_in)
        for nbg in self.modwave_layers[1:]:
            backcast, forecast_graph = nbg(history_in=forecast, node_id_in=node_id_in, time_of_day_in=time_of_day_in,
                                           day_in_week_in=day_in_week_in, wt1=wt1_in, wt2=wt2_in, wt3=wt3_in,
                                           wt4=wt4_in)
            forecast = forecast + forecast_graph
        forecast = forecast / self.hyperparams.num_stacks
        forecast = tf.where(tf.math.is_nan(forecast), tf.zeros_like(forecast), forecast)

        inputs = {'history': history_in, 'node_id': node_id_in,
                  'time_of_day': time_of_day_in, 'day_in_week': day_in_week_in,
                  'wt1': wt1_in, 'wt2': wt2_in, 'wt3': wt3_in, 'wt4': wt4_in}
        outputs = {'targets': forecast}
       
        return inputs, outputs




