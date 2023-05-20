import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import os
from datetime import datetime
import wandb
import random
import numpy as np
import ssl

from data.dataset_utils import load_basenji2
from models.model_utils import get_models

from config import get_config

ssl._create_default_https_context = ssl._create_unverified_context

random_state = get_config("general", "random_state")
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

def set_seed(seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(get_config("general","random_state"))
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"outputs/{now}")
    
    torch.multiprocessing.freeze_support()
    config_gen = get_config("general")

    if config_gen["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_gen["device"])

    num_epochs = config_gen["num_epochs"]
    scopes = get_config("data", "scopes")

    for model, config in get_models():
        os.makedirs(f"outputs/{now}/"+config["name"])
        wandb.init(project="CAE"+config["name"], config=get_config("wandb"))
        model = model.to(device)
        
        train_settings = config["train_settings"]
        loss_function = train_settings["loss_function"]
        optimizer = train_settings["optimizer"](model.parameters(), lr=train_settings["lr"])

        for epoch in range(num_epochs):
            trainloader, testloader, _ = load_basenji2()
            results = ""
            running_loss = 0.0
            print(f"Epoch: {epoch+1}")
            pbar = tqdm(total=len(trainloader))
            for i, data in enumerate(trainloader, 0):
                if i < len(trainloader):
                    x, y, organism = data
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x, organism)
                    loss = loss_function(outputs, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.update()
                else:
                    break
            pbar.close()
            
            # 各エポック後の処理
            with torch.no_grad():
                scores = {outer_k: {inner_k: 0.0 for inner_k in outer_v} for outer_k, outer_v in scopes.items()}
                pbar = tqdm(total=len(testloader))
                counter = 0
                for j, data in enumerate(testloader, 0):
                    if j < len(testloader):
                        x, y, organism = data
                        x, y = x.to(device), y.to(device)
                        pred = model(x, organism)
                        scores, count = config["train_settings"]["eval_function"](scopes, scores, pred, y, organism)
                        counter += count
                        pbar.update()
                    else:
                        break
                pbar.close()
            epoch_score = 0.0
            for outer_k, outer_v in scores.items():
                for inner_k, inner_v in outer_v.items():
                    scores[outer_k][inner_k] /= counter
                    epoch_score+=(inner_v/counter)
            epoch_score/=8
            epoch_loss= running_loss/(i+1)
            wandb.log({"Loss":epoch_loss, "Score":epoch_score})   
            result = f"Loss: {epoch_loss}\nEpoch_score: {epoch_score}\nScores:\n"
            for outer_k, outer_v in scores.items():
                result += f"  {outer_k}:\n"
                for inner_k in outer_v:
                    score = scores[outer_k][inner_k]
                    result += f"    {inner_k}: {score}\n"
            results += ("Epoch:"+str(epoch+1)+"  "+result)
            print(result)
            
        # モデル学習完了後の処理
        out_dir = f"outputs/{now}/"+config["name"]+"/"
        torch.save(model.state_dict(), out_dir+"model.pth")
        with open(out_dir+"results.txt", "w") as file:
            file.write(results)
        wandb.finish()
        print("Training finished")
        

if __name__ == "__main__":
    train()