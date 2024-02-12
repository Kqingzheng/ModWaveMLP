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
#         self.DLinear_node = DLinearLayer_node(hyperparams=hyperparams,
#                                                   input_size=self.input_size,
#                                                   output_size=hyperparams.horizon,
#                                                   num_nodes=self.num_nodes,
#                                                   name=f"DLinearLayer_node")
#
#
# def call(self, history_in, node_id_in, time_of_day_in, day_in_week_in, wt1, wt2, wt3, wt4, training=False):
#     forecast_node = self.DLinear_node(history_in=history_in, node_id_in=node_id_in,
#                                       time_of_day_in=time_of_day_in, day_in_week_in=day_in_week_in,
#                                       wt1=wt1, wt2=wt2, wt3=wt3, wt4=wt4)
#
#     return history_in,forecast_node


class DLinearLayer_node(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, num_nodes: int, **kw):
        super(DLinearLayer_node, self).__init__(**kw)
        self.kernel_size = 2
        self.output_steps = hyperparams.num_nodes
        self.output_features = output_size
        self.separate_features = output_size
        self.kernel_initializer = "he_normal"

        # self.built_input_shape = input_shape

        if self.separate_features:
            self.trend_dense = []
            self.residual_dense = []
            for feature in range(self.output_features):
                self.trend_dense.append(tf.keras.layers.Dense(self.output_steps,
                                                              kernel_initializer=self.kernel_initializer,
                                                              name="trend_decoder_feature_" + str(feature)))
                self.residual_dense.append(tf.keras.layers.Dense(self.output_steps,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 name="residual_decoder_feature_" + str(feature)))
        else:
            self.trend_dense = tf.keras.layers.Dense(self.output_steps * self.output_features,
                                                     kernel_initializer=self.kernel_initializer,
                                                     name="trend_recomposer")
            self.residual_dense = tf.keras.layers.Dense(self.output_steps * self.output_features,
                                                        kernel_initializer=self.kernel_initializer,
                                                        name="residual_recomposer")

    def call(self, history_in, node_id_in, time_of_day_in, day_in_week_in, wt1, wt2, wt3, wt4, training=False):
        # print(history_in.shape)
        # history_in = tf.keras.layers.Permute((2,1))(history_in)

        # print(history_in.shape)
        trend = tf.keras.layers.AveragePooling1D(pool_size=self.kernel_size,
                                                 strides=1,
                                                 padding="same",
                                                 name="trend_decomposer")(history_in)
        # print(trend.shape)

        residual = tf.keras.layers.Subtract(name="residual_decomposer")([history_in, trend])

        if self.separate_features:
            paths = []

            for feature in range(self.output_features):
                trend_sliced = tf.keras.layers.Lambda(lambda x: x[:, :, feature],
                                                      name="trend_slicer_feature_" + str(feature))(trend)
                trend_sliced = self.trend_dense[feature](trend_sliced)
                trend_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                       name="reshape_trend_feature_" + str(feature))(trend_sliced)

                residual_sliced = tf.keras.layers.Lambda(lambda x: x[:, :, feature],
                                                         name="residuals_slicer_feature_" + str(feature))(residual)
                residual_sliced = self.residual_dense[feature](residual_sliced)
                residual_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                          name="reshape_residual_feature_" + str(feature))(
                    residual_sliced)

                path = tf.keras.layers.Add(name="recomposer_feature_" + str(feature))([trend_sliced, residual_sliced])

                paths.append(path)

            reshape = tf.keras.layers.Concatenate(axis=2,
                                                  name="output_recomposer")(paths)
        else:
            flat_residual = tf.keras.layers.Flatten()(residual)
            flat_trend = tf.keras.layers.Flatten()(trend)

            residual = self.residual_dense(flat_residual)

            trend = self.trend_dense(flat_trend)

            add = tf.keras.layers.Add(name="recomposer")([residual, trend])

            reshape = tf.keras.layers.Reshape((self.output_steps, self.output_features))(add)

        # reshape = tf.keras.layers.Reshape((self.output_features,self.output_steps))(reshape)

        return reshape


class DLinearLayer_time(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, num_nodes: int, **kw):
        super(DLinearLayer_time, self).__init__(**kw)
        self.kernel_size = 6
        self.output_steps = output_size
        self.output_features = hyperparams.num_nodes
        self.separate_features = hyperparams.num_nodes
        self.kernel_initializer = "he_normal"

        # self.built_input_shape = input_shape

        if self.separate_features:
            self.trend_dense = []
            self.residual_dense = []
            for feature in range(self.output_features):
                self.trend_dense.append(tf.keras.layers.Dense(self.output_steps,
                                                              kernel_initializer=self.kernel_initializer,
                                                              name="trend_decoder_feature_" + str(feature)))
                self.residual_dense.append(tf.keras.layers.Dense(self.output_steps,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 name="residual_decoder_feature_" + str(feature)))
        else:
            self.trend_dense = tf.keras.layers.Dense(self.output_steps * self.output_features,
                                                     kernel_initializer=self.kernel_initializer,
                                                     name="trend_recomposer")
            self.residual_dense = tf.keras.layers.Dense(self.output_steps * self.output_features,
                                                        kernel_initializer=self.kernel_initializer,
                                                        name="residual_recomposer")

    def call(self, history_in, node_id_in, time_of_day_in, day_in_week_in, wt1, wt2, wt3, wt4, training=False):
        # print(history_in.shape)
        history_in = tf.keras.layers.Permute((2, 1))(history_in)

        # print(history_in.shape)
        trend = tf.keras.layers.AveragePooling1D(pool_size=self.kernel_size,
                                                 strides=1,
                                                 padding="same",
                                                 name="trend_decomposer")(history_in)
        # print(trend.shape)

        residual = tf.keras.layers.Subtract(name="residual_decomposer")([history_in, trend])

        if self.separate_features:
            paths = []

            for feature in range(self.output_features):
                trend_sliced = tf.keras.layers.Lambda(lambda x: x[:, :, feature],
                                                      name="trend_slicer_feature_" + str(feature))(trend)
                trend_sliced = self.trend_dense[feature](trend_sliced)
                trend_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                       name="reshape_trend_feature_" + str(feature))(trend_sliced)

                residual_sliced = tf.keras.layers.Lambda(lambda x: x[:, :, feature],
                                                         name="residuals_slicer_feature_" + str(feature))(residual)
                residual_sliced = self.residual_dense[feature](residual_sliced)
                residual_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                          name="reshape_residual_feature_" + str(feature))(
                    residual_sliced)

                path = tf.keras.layers.Add(name="recomposer_feature_" + str(feature))([trend_sliced, residual_sliced])

                paths.append(path)

            reshape = tf.keras.layers.Concatenate(axis=2,
                                                  name="output_recomposer")(paths)
        else:
            flat_residual = tf.keras.layers.Flatten()(residual)
            flat_trend = tf.keras.layers.Flatten()(trend)

            residual = self.residual_dense(flat_residual)

            trend = self.trend_dense(flat_trend)

            add = tf.keras.layers.Add(name="recomposer")([residual, trend])

            reshape = tf.keras.layers.Reshape((self.output_steps, self.output_features))(add)

        reshape = tf.keras.layers.Reshape((self.output_features, self.output_steps))(reshape)

        return reshape