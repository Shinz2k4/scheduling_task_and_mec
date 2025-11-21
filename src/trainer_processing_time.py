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
import joblib
import os, io, time, argparse
from utils.utils_func import get_active_service

N_SERVER = 4

def encode_model(result_str):
    try:
        result_dict = ast.literal_eval(result_str)
        model = result_dict.get("payload", {}).get("model", "")
        model_id = 0
        if model =="ssd":
            model_id = 9
        elif model =="resnet34":
            model_id = 3
    except:
        return -1
    return model_id

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


class DelayPredictor(nn.Module):
    def __init__(self, input_dim=10):
        super(DelayPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Softplus()  
        )

    def forward(self, x):
        return self.net(x)
    
    def save_model(self, path="path.pth"):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path="path.pth"):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", default="10")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    list_docker, list_service_in_docker = get_active_service()

    df = pd.read_csv(f"/home/ailab/Documents/dminh/mec_simulation_mutipleserver/output_file3/results_n_server_4_n_user_10_random_seed_45.csv")
    # df = pd.read_csv(f"./../output_file_{len(list_service_in_docker[1])}_service_{args.n_users}_users/results_n_server_4_n_user_{args.n_users}_random_seed_{args.seed}.csv")
    print(f"len df {len(df)}")
    df["model_code"] = df["results"].apply(encode_model)
    df["server_index"] = df["results"].apply(extract_server_port)
    df["state_list"] = df["current_state_information"].apply(parse_state_info)

    vectors = []
    total_delays = []

    for _, row in df.iterrows():
        base_values = [row["id_picture"], row["model_code"], row["server_index"] + 1] 
        state_values = row["state_list"]
        
        if len(state_values) % N_SERVER != 0:
            raise ValueError(f"Số phần tử trong state_list không chia hết cho 4 (len={len(state_values)})")
        
        chunk_size = len(state_values) // N_SERVER
        start = (row["server_index"]-1) * chunk_size
        end = start + chunk_size
        selected_state = state_values[start:end]
        
        vectors.append(base_values + selected_state)
        total_delays.append(row["total_delay"])

    X = np.array(vectors, dtype=float)
    y = np.array(total_delays, dtype=float)
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    y_scaled = y.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

        
    model = DelayPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    batch_size = 512 

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 4000
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)  
        
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")



    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f"Test MSE Loss: {test_loss.item():.6f}")

    save_dir = "train_result"
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(f"{save_dir}/pretrained_processing_estimation.pth")
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss qua Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_pred.max()], [y_test.min(), y_pred.max()], 'r--', label="Ideal")
    plt.xlabel("Actual total_delay")
    plt.ylabel("Predicted total_delay")
    plt.title("Actual vs Predicted total_delay")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted_scatter.png"))
    plt.close()

    errors = y_test - y_pred
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=30, color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "prediction_error_histogram.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='orange', linewidth=2, alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("total_delay")
    plt.title("Actual vs Predicted total_delay (Line Chart)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted_line.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(range(len(errors)), errors, label='Prediction Error', color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Error (Predicted - Actual)")
    plt.title("Prediction Error Across Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "prediction_error_line.png"))
    plt.close()

    print(f"Tất cả hình đã lưu vào thư mục: {save_dir}")