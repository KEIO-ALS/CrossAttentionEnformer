import torch
import torch.nn as nn
import torch.optim as optim

from models.validation_functions import poisson_nll_loss, get_correlation_coefficient

c = {
    "general":{
        "num_epochs": 151,
        "random_state": 111,
        "batch_size": 5,
        "num_workers": 1,
        "device": "cpu",
    },
    "data":{
        "metadata_human":{
            "num_targets": 5313,
            # "train_seqs": 34021,
            # "valid_seqs": 2213,
            # "test_seqs": 1937,
            "train_seqs": 20,
            "valid_seqs": 2213,
            "test_seqs": 10,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        },
        "metadata_mouse":{
            "num_targets": 1643,
            # "train_seqs": 29295,
            # "valid_seqs": 2209,
            # "test_seqs": 2017,
            "train_seqs": 20,
            "valid_seqs": 2209,
            "test_seqs": 10,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        },
        "scopes":{
            "human":{
                "DNase":(0, 673),
                "ATAC":(674, 683),
                "ChIP":(684, 4674),
                "CAGE":(4675, 5312),
            },
            "mouse":{
                "DNase":(0, 100),
                "ATAC":(101, 227),
                "ChIP":(228, 1285),
                "CAGE":(1286, 1642),
            },
        }
    },
    "models":{
        "Enformer":{
            "name": "Enformer",
            "state": True,
            "train_settings":{
                "loss_function": poisson_nll_loss,
                "optimizer": optim.Adam,
                "lr": 5e-4,
                "eval_function": get_correlation_coefficient,
            },
            "param":{
                "channels": 1024,
                "num_conv": 6,
                "num_attn": 11,
                "dropout_attn": .4,
                "dropout_output": .05,
            },        
        },
    },
    "wandb":{
        "learning_rate": 0.02,
        "architecture": "Enformer",
        "dataset": "Basenji2",
        "epochs": 10,
    },
}

def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config