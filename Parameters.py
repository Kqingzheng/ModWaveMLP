from typing import Dict, NamedTuple

hyperparams_defaults = {
    "dataset": "metr-la",
    "repeat": list(range(2)),
    "epochs": [80],
    "steps_per_epoch": [800],  # 800 METR-LA, 800 PEMS-BAY
    "block_layers": 3,
    "hidden_units": 128,
    "blocks": 2,  
    "horizon": 12,
    "history_length": 12,
    "init_learning_rate": 1e-3,
    "decay_steps": 4,
    "decay_rate": 0.5,
    "batch_size": 4,
    "weight_decay": 1e-5,
    "node_id_dim": 96,
    "num_nodes": 325,  # 207 | 325
    "num_stacks": 4,
    
}


class Parameters(NamedTuple):
    dataset: str
    repeat: int
    epochs: int
    steps_per_epoch: int
    block_layers: int
    hidden_units: int
    blocks: int
    horizon: int
    history_length: int
    init_learning_rate: float
    decay_steps: int
    decay_rate: float
    batch_size: int
    weight_decay: float
    node_id_dim: int
    num_nodes: int
    num_stacks: int
    