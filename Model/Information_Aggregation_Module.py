import tensorflow as tf
from Parameters import Parameters
epsilon=10
def Information_Aggregation_Module(hyperparams: Parameters, node_embeddings, history_in, node_id):
    node_embeddings_dp = tf.tensordot(node_embeddings, tf.transpose(node_embeddings, perm=[1, 0]), axes=1)
    node_embeddings_dp = tf.math.exp(epsilon * node_embeddings_dp)
    node_embeddings_dp = node_embeddings_dp[tf.newaxis, :, :, tf.newaxis]

    level = tf.reduce_max(history_in, axis=-1, keepdims=True)

    history = tf.math.divide_no_nan(history_in, level)
    shape = history_in.get_shape().as_list()
    all_node_history = tf.tile(history_in[:, tf.newaxis, :, :], multiples=[1, hyperparams.num_nodes, 1, 1])

    all_node_history = all_node_history * node_embeddings_dp
    all_node_history = tf.reshape(all_node_history, shape=[-1, hyperparams.num_nodes, hyperparams.num_nodes * shape[2]])
    all_node_history = tf.math.divide_no_nan(all_node_history - level, level)
    all_node_history = tf.where(all_node_history > 0, all_node_history, 0.0)

    history = tf.concat([history, all_node_history], axis=-1)

    history = tf.concat([history, node_id], axis=-1)
    return history,level
