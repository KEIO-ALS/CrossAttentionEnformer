import torch
import pandas as pd

def poisson_nll_loss(pred, y):
    pred, y = pred.float(), y.float()
    loss = pred - y * torch.log(pred)
    return loss.mean()

def _torch2numpy(tensor):
    return tensor.to("cpu").detach().numpy().copy()

def get_corr(p, t):
    s1 = pd.Series(_torch2numpy(p).flatten())
    s2 = pd.Series(_torch2numpy(t).flatten())
    return s1.corr(s2)


def get_correlation_coefficient(scopes, scores, pred, y, organism):
    counter = 0
    for i in range(pred.shape[0]):
        for scope_key in scopes[organism]:
            scope = scopes[organism][scope_key]
            p = pred[i, :, scope[0]:scope[1]+1]
            t = y[i, :, scope[0]:scope[1]+1]
            scores[organism][scope_key] += get_corr(p, t)
        counter+=1
    return scores, counter
