import tensorflow as tf
from Parameters import Parameters
from Model.MDLBlock import MDLBlock
from Model.TimeGate import TimeGate
from Model.Information_Aggregation_Module import Information_Aggregation_Module

class Mode_Information_Coding_Block(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, num_nodes: int, time_gate_backcast_size,**kw):
        super(Mode_Information_Coding_Block, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.num_nodes = num_nodes
        self.input_size = input_size
       
        self.mdlblock = MDLBlock(hyperparams=hyperparams, input_size=input_size,name="mdlblock")

        self.node_id_em = tf.keras.layers.Embedding(input_dim=self.num_nodes,
                                                    output_dim=self.hyperparams.node_id_dim,
                                                    embeddings_initializer='uniform',
                                                    input_length=self.num_nodes, name="dept_id_em",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(
                                                        hyperparams.weight_decay))

        self.TimeGate = TimeGate(hyperparams=hyperparams, time_gate_backcast_size=time_gate_backcast_size,name="TimeGate")





    def call(self, history_in, node_id_in, time_of_day_in, training=False):
        node_id = self.node_id_em(node_id_in)

        node_embeddings = tf.squeeze(node_id[0, :, :])
        node_id = tf.squeeze(node_id, axis=-2)
       
        time_gate_forward, time_gate_backward = self.TimeGate(node_id, time_of_day_in)
 
        history_in = history_in / (1.0 + time_gate_backward)
        history,level = Information_Aggregation_Module(self.hyperparams, node_embeddings,history_in, node_id)
        backcast, forecast_out = self.mdlblock(history)
        forecast = forecast_out * level
        forecast = (1.0 + time_gate_forward) * forecast
      
        return backcast, forecast