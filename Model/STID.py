import tensorflow as tf

from Parameters import Parameters
# # The model is used as follows:
# class MoDWaveLayer(tf.keras.layers.Layer):
#     def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, num_nodes: int, **kw):
#         super(MoDWaveLayer, self).__init__(**kw)
#         self.hyperparams = hyperparams
#         self.num_nodes = num_nodes
#         self.input_size = input_size
#
#         self.STID = STID(hyperparams=hyperparams)
#
# def call(self, history_in, node_id_in, time_of_day_in, day_in_week_in, wt1, wt2, wt3, wt4, training=False):
#     return history_in,self.STID(history_in=history_in,time_of_day_in=time_of_day_in,day_in_week_in=day_in_week_in,node_id_in=node_id_in)


class MultiLayerPerceptron(tf.keras.layers.Layer):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), use_bias=True)
        self.fc2 = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), use_bias=True)
        self.act = tf.keras.layers.Activation('relu')
        self.drop = tf.keras.layers.Dropout(rate=0.15)

    def call(self, input_data):
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = tf.keras.layers.Add()([hidden, input_data])  # residual
        return hidden



class STID(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, **model_args):
        super(STID, self).__init__()
        self.emdim = 32
        # attributes
        self.hyperparams = hyperparams
        self.num_nodes = hyperparams.num_nodes
        self.node_dim = self.emdim
        self.input_len = hyperparams.history_length
        self.embed_dim = self.emdim
        self.output_len = hyperparams.history_length
        self.num_layer = 3
        self.temp_dim_tid = self.emdim
        self.temp_dim_diw = self.emdim
        self.time_of_day_size = 288
        self.day_of_week_size = 7

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = tf.keras.layers.Embedding(input_dim=self.num_nodes,
                                                      output_dim=self.node_dim,
                                                      embeddings_initializer='uniform',
                                                      input_length=self.num_nodes, name="dept_id_em",
                                                      embeddings_regularizer=tf.keras.regularizers.l2(
                                                          hyperparams.weight_decay))

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = tf.keras.layers.Embedding(input_dim=self.time_of_day_size,
                                                             output_dim=self.temp_dim_tid,
                                                             embeddings_initializer='uniform',
                                                             input_length=self.time_of_day_size, name="dept_tid_em",
                                                             embeddings_regularizer=tf.keras.regularizers.l2(
                                                                 hyperparams.weight_decay))
        if self.if_day_in_week:

            self.day_in_week_emb = tf.keras.layers.Embedding(input_dim=self.day_of_week_size,
                                                             output_dim=self.temp_dim_diw,
                                                             embeddings_initializer='uniform',
                                                             input_length=self.day_of_week_size, name="dept_diw_em",
                                                             embeddings_regularizer=tf.keras.regularizers.l2(
                                                                 hyperparams.weight_decay))

        # embedding layer
        self.time_series_emb_layer = tf.keras.layers.Conv2D(self.embed_dim, kernel_size=(1, 1), use_bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + self.temp_dim_tid * int(
            self.if_time_in_day) + self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = [MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]

        # regression
        self.regression_layer = tf.keras.layers.Conv2D(self.output_len, kernel_size=(1, 1), use_bias=True)

    def call(self, history_in, time_of_day_in, day_in_week_in, node_id_in, **kwargs):
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        # print(history_in.shape)
        input_data = history_in
        # print(input_data.shape)

        if self.if_time_in_day:
            time_of_day_in = tf.transpose(time_of_day_in, [0, 2, 1])
            t_i_d_data = time_of_day_in
            time_in_day_emb = self.time_in_day_emb(t_i_d_data[:, -1, :] * self.time_of_day_size)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            day_in_week_in = tf.transpose(day_in_week_in, [0, 2, 1])
            d_i_w_data = day_in_week_in
            day_in_week_emb = self.day_in_week_emb(d_i_w_data[:, -1, :])
        else:
            day_in_week_emb = None

        # time series embedding
        # batch_size, num_nodes, _= input_data.shape
        # print(batch_size)
        # print(num_nodes)
        # input_data = tf.transpose(input_data, [0, 2, 1, 3])
        # input_data = tf.reshape(input_data, [batch_size, num_nodes, -1, 1])
        input_data = input_data[:, :, tf.newaxis, :]
        time_series_emb = self.time_series_emb_layer(input_data)
        time_series_emb = tf.transpose(time_series_emb, [0, 3, 1, 2])
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_id = self.node_emb(node_id_in)
            # node_embeddings = tf.squeeze(node_id[0, :, :])
            node_id = tf.squeeze(node_id, axis=-2)
            # print(node_id.shape)
            node_emb.append(tf.expand_dims(tf.transpose(node_id, [0, 2, 1]), -1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(tf.transpose(time_in_day_emb, [0, 2, 1])[:, :, :, tf.newaxis])
        if day_in_week_emb is not None:
            tem_emb.append(tf.transpose(day_in_week_emb, [0, 2, 1])[:, :, :, tf.newaxis])

        # concatenate all embeddings
        hidden = tf.concat([time_series_emb] + node_emb + tem_emb, axis=1)
        hidden = tf.transpose(hidden, [0, 2, 3, 1])
        # encoding
        for layer in self.encoder:
            hidden = layer(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        prediction = tf.reshape(prediction, [-1, self.hyperparams.num_nodes, self.hyperparams.history_length])
        return prediction