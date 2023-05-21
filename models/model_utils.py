from models.Enformer import Enformer
from models.CrossAttentionEnformer import CrossAttentionEnformer

import sys
sys.path.append('../')
from config import get_config

models = {
    "Enformer": Enformer,
    "CrossAttentionEnformer": CrossAttentionEnformer,
}     

def get_models():
    selected_models = []
    for key in models:
        model_config = get_config("models")[key]
        if model_config["state"]:
            selected_models.append([models[key](model_config["param"]), model_config])
    return selected_models