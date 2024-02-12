import os
import numpy as np
import tensorflow as tf
from Model.MoDWaveMLP import MoDWaveMLP
from Parameters import Parameters
from utils import MetricsCallback
from itertools import product

class Trainer:
    def __init__(self, hyperparams: Parameters, logdir: str):
        inp = dict(hyperparams._asdict())
        values = [v if isinstance(v, list) else [v] for v in inp.values()]
        self.hyperparams = [Parameters(**dict(zip(inp.keys(), v))) for v in product(*values)]
        inp_lists = {k: v for k, v in inp.items() if isinstance(v, list)}
        values = [v for v in inp_lists.values()]
        variable_values = [dict(zip(inp_lists.keys(), v)) for v in product(*values)]
        folder_names = []
        for d in variable_values:
            folder_names.append(
                ';'.join(['%s=%s' % (key, value) for (key, value) in d.items()])
            )
        self.history = []
        self.forecasts = []
        self.models = []
        self.logdir = logdir
        self.folder_names = folder_names
        for i, h in enumerate(self.hyperparams):
            self.models.append(MoDWaveMLP(hyperparams=h, name=f"modwave_model_{i}",
                                      logdir=os.path.join(self.logdir, folder_names[i]),
                                      num_nodes=h.num_nodes))

    def generator(self, ds, hyperparams: Parameters):
        while True:
            batch = ds.get_batch(batch_size=hyperparams.batch_size)
            weights = np.all(batch["y"] > 0, axis=-1, keepdims=False).astype(np.float32)
            weights = weights / np.prod(weights.shape)
            yield {"history": batch["x"][..., 0], "node_id": batch["node_id"], "time_of_day": batch["x"][..., 1],
                   "day_in_week": batch["x"][..., 2], 'wt1': batch["x"][..., 3], 'wt2': batch["x"][..., 4],
                   'wt3': batch["x"][..., 5], 'wt4': batch["x"][..., 6]}, \
                  {"targets": batch["y"]}, \
                  weights
    

    def fit(self, dataset, verbose=1):
        for i, hyperparams in enumerate(self.hyperparams):
            if verbose > 0:
                print(f"Fitting model {i + 1} out of {len(self.hyperparams)}, {self.folder_names[i]}")

            boundary_step = hyperparams.epochs // 10
            boundary_start = hyperparams.epochs - boundary_step * hyperparams.decay_steps - 1

            boundaries = list(range(boundary_start, hyperparams.epochs, boundary_step))
            values = list(hyperparams.init_learning_rate * hyperparams.decay_rate ** np.arange(0, len(boundaries) + 1))
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)

            lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

            metrics = MetricsCallback(dataset=dataset, logdir=self.models[i].logdir)
            tb = tf.keras.callbacks.TensorBoard(log_dir=self.models[i].logdir, embeddings_freq=10)

            self.models[i].model.compile(optimizer=tf.keras.optimizers.Adam(),
                                         loss={"targets": tf.keras.losses.MeanAbsoluteError(
                                             reduction=tf.keras.losses.Reduction.SUM)},
                                         loss_weights={"targets": 1.0})
            self.models[i].model.summary()
            fit_output = self.models[i].model.fit(self.generator(ds=dataset, hyperparams=hyperparams),
                                                  callbacks=[lr, metrics],  # tb
                                                  epochs=hyperparams.epochs,
                                                  steps_per_epoch=hyperparams.steps_per_epoch,
                                                  verbose=verbose)
            self.history.append(fit_output.history)
