import pandas as pd
import ast
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
N_SERVER = 4

def encode_model(result_str):
    try:
        result_dict = ast.literal_eval(result_str)
        model = result_dict.get("payload", {}).get("model", "")
        if model == "ssd":
            return 9
        elif model == "resnet18":
            return 0
        else:
            return -1
    except:
        return -1

def extract_server_port(result_str):
    try:
        result_dict = ast.literal_eval(result_str)
        backend = result_dict.get("backend", "")
        match = re.search(r":(\d+)/", backend)
        if match:
            port = int(match.group(1))-1
            last_three = port % 1000 
            ser_id = last_three // 100 -1 
            return ser_id
        else:
            return -1
    except:
        return -1

def parse_state_info(state_str):
    try:
        return list(map(float, ast.literal_eval(state_str)))
    except:
        return []


class DPredictor(nn.Module):
    def __init__(self, input_dim=10):
        super(DPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.01),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.01),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Softplus()  # đảm bảo đầu ra không âm
        )

    def forward(self, x):
        return self.net(x)
    
    def save_model(self, path="path.pth"):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path="path.pth"):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
