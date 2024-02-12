import copy
import tensorflow as tf
from Parameters import Parameters
from Model.MDLBlock import MDLBlock
from Model.TimeGate import TimeGate
from Model.Information_Aggregation_Module import Information_Aggregation_Module
from Model.Mode_Information_Coding_Block import Mode_Information_Coding_Block

class MoDWaveLayer(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, num_nodes: int, **kw):
        super(MoDWaveLayer, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.num_nodes = num_nodes
        self.input_size = input_size
    
        self.hyperparams_patch = copy.deepcopy(hyperparams)
        self.hyperparams_patch = self.hyperparams_patch._replace(horizon = int(hyperparams.horizon/2))
        self.hyperparams_patch = self.hyperparams_patch._replace(history_length = int(hyperparams.history_length/2))
       
        self.hyperparams_patch34 = copy.deepcopy(hyperparams)
        self.hyperparams_patch34 = self.hyperparams_patch34._replace(horizon = int(hyperparams.horizon/4))
        self.hyperparams_patch34 = self.hyperparams_patch34._replace(history_length = int(hyperparams.history_length/4))

        self.day_mdlblock = MDLBlock(hyperparams=hyperparams, input_size=input_size,name="day_mdlblock")
        self.week_mdlblock = MDLBlock(hyperparams=hyperparams, input_size=input_size,name="week_mdlblock")
        self.day_add_mdlblock = MDLBlock(hyperparams=hyperparams, input_size=input_size,name="day_add_mdlblock")
        self.week_add_mdlblock = MDLBlock(hyperparams=hyperparams, input_size=input_size,name="day_week_mdlblock")
        self.final_mdlblock = MDLBlock(hyperparams=hyperparams, input_size=self.hyperparams.history_length * 6+6,name="final_mdlblock")
        
        self.node_id_em = tf.keras.layers.Embedding(input_dim=self.num_nodes,
                                                    output_dim=self.hyperparams.node_id_dim,
                                                    embeddings_initializer='uniform',
                                                    input_length=self.num_nodes, name="dept_id_em",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(
                                                        hyperparams.weight_decay))
    
   
        self.DayGate = TimeGate(hyperparams=hyperparams, time_gate_backcast_size=hyperparams.history_length,name="DayGate")
        self.WeekGate = TimeGate(hyperparams=hyperparams, time_gate_backcast_size=hyperparams.history_length,name="WeekGate")

        
        self.wt1_history = Mode_Information_Coding_Block(hyperparams=hyperparams,
                                                  input_size=hyperparams.horizon*4*hyperparams.num_nodes + hyperparams.node_id_dim + hyperparams.history_length*4,
                                                  output_size=hyperparams.horizon,
                                                  num_nodes=self.num_nodes,
                                                  time_gate_backcast_size=hyperparams.history_length*4,
                                                  name=f"wavelet")
  
        self.patch1_history = Mode_Information_Coding_Block(hyperparams=self.hyperparams_patch,
                                                  input_size=self.hyperparams_patch.horizon*self.hyperparams_patch.num_nodes + self.hyperparams_patch.node_id_dim + self.hyperparams_patch.history_length,
                                                  output_size=self.hyperparams_patch.horizon,
                                                  num_nodes=self.num_nodes,
                                                  time_gate_backcast_size=self.hyperparams_patch.history_length,
                                                  name=f"patch1")
        self.patch2_history = Mode_Information_Coding_Block(hyperparams=self.hyperparams_patch,
                                                  input_size=self.hyperparams_patch.horizon*self.hyperparams_patch.num_nodes + self.hyperparams_patch.node_id_dim + self.hyperparams_patch.history_length,
                                                  output_size=self.hyperparams_patch.horizon,
                                                  num_nodes=self.num_nodes,
                                                  time_gate_backcast_size=self.hyperparams_patch.history_length,
                                                  name=f"patch2")
      
        self.patch3_history = Mode_Information_Coding_Block(hyperparams=self.hyperparams_patch34,
                                                  input_size=self.hyperparams_patch34.horizon*self.hyperparams_patch34.num_nodes + self.hyperparams_patch34.node_id_dim + self.hyperparams_patch34.history_length,
                                                  output_size=self.hyperparams_patch34.horizon,
                                                  num_nodes=self.num_nodes,
                                                  time_gate_backcast_size=self.hyperparams_patch34.history_length,
                                                  name=f"patch3")
        self.patch4_history = Mode_Information_Coding_Block(hyperparams=self.hyperparams_patch34,
                                                  input_size=self.hyperparams_patch34.horizon*self.hyperparams_patch34.num_nodes + self.hyperparams_patch34.node_id_dim + self.hyperparams_patch34.history_length,
                                                  output_size=self.hyperparams_patch34.horizon,
                                                  num_nodes=self.num_nodes,
                                                  time_gate_backcast_size=self.hyperparams_patch34.history_length,
                                                  name=f"patch4")
        

    def call(self, history_in, node_id_in, time_of_day_in, day_in_week_in, wt1, wt2, wt3, wt4, training=False):
        node_id = self.node_id_em(node_id_in)

        node_embeddings = tf.squeeze(node_id[0, :, :])
        node_id = tf.squeeze(node_id, axis=-2)
        
        time_gate_forward, time_gate_backward = self.DayGate(node_id, time_of_day_in)
        week_gate_forward, week_gate_backward = self.WeekGate(node_id, day_in_week_in)
        history_in = history_in / (1.0 + time_gate_backward)
        history_in_week = history_in / (1.0 + week_gate_backward)
        
        history,level = Information_Aggregation_Module(self.hyperparams, node_embeddings,history_in, node_id)
        history_week,level_week = Information_Aggregation_Module(self.hyperparams, node_embeddings,history_in_week, node_id)
       
        backcast, forecast_out = self.day_mdlblock(history)
        forecast = forecast_out * level
        forecast = (1.0 + time_gate_forward) * forecast


        backcast_week, forecast_out_week = self.week_mdlblock(history_week)

        forecast_week = forecast_out_week * level_week
        forecast_week = (1.0 + week_gate_forward) * forecast_week



        backcast_add, forecast_out_add = self.day_add_mdlblock(history)

        
        forecast_add = forecast_out_add * level
        forecast_add = (1.0 + time_gate_forward) * forecast_add

        backcast_week_add, forecast_out_week_add = self.week_add_mdlblock(history_week)

       
        forecast_week_add = forecast_out_week_add * level_week
        forecast_week_add = (1.0 + week_gate_forward) * forecast_week_add


      
        wt_12 = tf.concat([wt1, wt2], axis=-1)
        wt_123 = tf.concat([wt_12, wt3], axis=-1)
        wt_final = tf.concat([wt_123, wt4], axis=-1)
     
        wt1_backcast, wt1_forecast = self.wt1_history(history_in=wt_final, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
        

  
        patch1_backcast, patch1_forecast = self.patch1_history(history_in=history_in[:, :, :self.hyperparams_patch.horizon], node_id_in=node_id_in, time_of_day_in=time_of_day_in[:, :, :self.hyperparams_patch.horizon])
        patch2_backcast, patch2_forecast = self.patch2_history(history_in=history_in[:, :, self.hyperparams_patch.horizon:], node_id_in=node_id_in, time_of_day_in=time_of_day_in[:, :, self.hyperparams_patch.horizon:])
        patch12_forecast = tf.concat([patch1_forecast,patch2_forecast],axis=-1)

 
       
        patch3_backcast, patch3_forecast = self.patch3_history(history_in=history_in[:, :, :self.hyperparams_patch34.horizon], node_id_in=node_id_in, time_of_day_in=time_of_day_in[:, :, :self.hyperparams_patch34.horizon])
        patch4_backcast, patch4_forecast = self.patch4_history(history_in=history_in[:, :, -self.hyperparams_patch34.horizon:], node_id_in=node_id_in, time_of_day_in=time_of_day_in[:, :, -self.hyperparams_patch34.horizon:])
        patch34_forecast = tf.concat([patch3_forecast,patch4_forecast],axis=-1)
        patch_forecast = tf.concat([patch12_forecast,patch34_forecast],axis=-1)
       


        forecast_time = tf.concat([forecast, forecast_week], axis=-1)
        forecast_add = tf.concat([forecast_add, forecast_week_add], axis=-1)
        
        forecast_time = tf.concat([forecast_time, forecast_add], axis=-1)
        forecast = tf.concat([forecast_time,  wt1_forecast], axis=-1)

        forecast = tf.concat([forecast, patch_forecast], axis=-1)

        backcast_final, forecast_out_final = self.final_mdlblock(forecast)


        backcast = backcast_final
        forecast = forecast_out_final

        return backcast, forecast