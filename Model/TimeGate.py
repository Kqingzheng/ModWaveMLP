import tensorflow as tf
from Parameters import Parameters


class TimeGate(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, time_gate_backcast_size, **kw):
        super(TimeGate, self).__init__(**kw)
        self.time_gate1 = tf.keras.layers.Dense(hyperparams.hidden_units,
                                                activation=tf.keras.activations.relu,
                                                name=f"time_gate1")
        self.time_gate2 = tf.keras.layers.Dense(hyperparams.history_length,
                                                activation=None,
                                                name=f"time_gate2")
        self.time_gate3 = tf.keras.layers.Dense(time_gate_backcast_size,
                                                activation=None,
                                                name=f"time_gate3")


    def call(self, node_id, time_of_day_in):
         time_gate = self.time_gate1(tf.concat([node_id, time_of_day_in], axis=-1))
         time_gate_forward = self.time_gate2(time_gate)
         time_gate_backward = self.time_gate3(time_gate)
         return time_gate_forward, time_gate_backward