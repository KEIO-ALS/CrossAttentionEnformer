import torch
import torch.nn as nn
import torch.optim as optim

from models.validation_functions import get_classification_accuracy

c = {
    "general":{
        "num_epochs": 2,
        "random_state": 111,
        "batch_size": 100,
        "num_workers": 1,
        "device": "cpu",
    },
    "data":{
        "metadata_human":{
            "num_targets": 5313,
            "train_seqs": 34021,
            "valid_seqs": 2213,
            "test_seqs": 1937,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        },
        "metadata_mouse":{
            "num_targets": 1643,
            "train_seqs": 29295,
            "valid_seqs": 2209,
            "test_seqs": 2017,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        }
    },
    "models":{
        "Enformer":{
            "name": "Enformer",
            "state": True,
            "train_settings":{
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param":{
                "channels": 1024,
            },        
        },
    },
    "wandb":{
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
}

def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config 