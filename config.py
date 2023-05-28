import torch
import torch.nn as nn
import torch.optim as optim

from models.validation_functions import poisson_nll_loss, get_correlation_coefficient

c = {
    "general":{
        "num_epochs": 1,
        "random_state": 111,
        "batch_size": 1,
        "num_workers": 1,
        "device": "cuda",
        "test_run": True,
    },
    "data":{
        "metadata_human":{
            "num_targets": 5313,
            "train_seqs": 34021,
            "test_seqs": 1937,
            "valid_seqs": 2213,
            # "train_seqs": 100,
            # "test_seqs": 100,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        },
        "metadata_mouse":{
            "num_targets": 1643,
            "train_seqs": 29295,
            "test_seqs": 2017,
            "valid_seqs": 2209,
            # "train_seqs": 100,
            # "test_seqs": 100,
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
                "channels": 1056,
                "num_conv": 6,
                "num_attn": 11,
                "dropout_attn": .05,
                "dropout_output": .05,
            },        
        },
        "CrossAttentionEnformer":{
            "name": "CrossAttentionEnformer",
            "state": True,
            "train_settings":{
                "loss_function": poisson_nll_loss,
                "optimizer": optim.Adam,
                "lr": 5e-4,
                "eval_function": get_correlation_coefficient,
            },
            "param":{
                "token_dim": 24,
                "num_latents": 1024,
                "channels": 1536,
                "num_attn": 11,
                "dropout_attn": .05,
                "dropout_output": .05,
            },        
        },
    },
    "wandb":{
        "learning_rate": 5e-4,
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