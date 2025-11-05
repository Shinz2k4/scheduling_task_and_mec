import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import ast
raw_path = 'experiments/'
save_path = 'experiments/analysis_result/'

directories = [name for name in os.listdir(raw_path)
            if os.path.isdir(os.path.join(raw_path, name))]

os.makedirs(save_path, exist_ok=True)

import json

def add_server_service(df: pd.DataFrame) -> pd.DataFrame:
    server_ids = []
    service_ids = []
    for res_str in df['results']:
        try:
            res = ast.literal_eval(res_str)
        except:
            res = {}
        backend = res.get('backend', '')
        if backend:
            try:
                port = int(backend.split(":")[-1].split("/")[0])
                last_three = port % 1000
                server_id = (last_three // 100) - 1
            except:
                server_id = None
        else:
            server_id = None
        server_ids.append(server_id)
        model = res.get('payload', {}).get('model', '')
        if model == "ssd":
            service_id = 9
        elif model == "resnet34":
            service_id = 3
        else:
            service_id = 0
        service_ids.append(service_id)
    df['server_id'] = server_ids
    df['service'] = service_ids
    return df

def draw_server_service_distribution(df: pd.DataFrame, title: str):
    min_data_sets = min(int(len(df)/2), 300)
    data_draw = df.iloc[min_data_sets-50:min_data_sets+50, :]

    plt.figure(figsize=(10, 5))
    plt.step(data_draw['task_id'], data_draw['server_id'], where='mid', linewidth=2)

    prev_id = None
    n_server = len(df['server_id'].unique()) 
    for task_id, server_id, status in zip(data_draw['task_id'], data_draw['server_id'], data_draw['current_state_information']):
        if prev_id is not None and server_id != prev_id:
            status = ast.literal_eval(status)
            length_data_status_in_one_servers=int(len(status)/n_server)
            status_server_id = status[(server_id-1)*length_data_status_in_one_servers:(server_id)*length_data_status_in_one_servers]
            status_prev_id = status[(prev_id-1)*length_data_status_in_one_servers:(prev_id)*length_data_status_in_one_servers]
            text_write = f"{prev_id}, {status_prev_id[0]}, {status_prev_id[6]}->{server_id}, {status_server_id[0]}, {status_server_id[6]}"
            plt.text(task_id, server_id + 0.1, text_write,
                    fontsize=8, color='red', rotation=45, ha='left', va='bottom')
        prev_id = server_id

    plt.xlabel('Task ID')
    plt.ylabel('Server ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    title_ = title.replace("/","/server_distribtion_").replace(".csv",".png")
    save_path_ = os.path.join(save_path, title_.replace(' ', '_'))
    plt.savefig(save_path_, dpi=300)
    plt.close()
    
def histogram(df: pd.DataFrame, column: str, title: str, path:str):
    data = df[column].dropna()  
    stats = {
        "count": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "25%": data.quantile(0.25),
        "75%": data.quantile(0.75),
        "var": data.var(), 
    }
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    textstr = "\n".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()])
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment="top", horizontalalignment="right",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7))
    plt.tight_layout()
    plt.tight_layout()
    path_ = path.replace("/",f"/histogram_of_{column}_").replace(".csv",".png")
    save_path_ = os.path.join(save_path, path_.replace(' ', '_'))
    plt.savefig(save_path_, dpi=300)
    plt.close()
def queueing_time_distribution(df: pd.DataFrame, title: str):
    service_mapping = {0: "mobilenet_v2", 3: "resnet34", 9: "ssd"}

    server = df['server_id'].unique()
    for s in server:
        server_df = df[df['server_id'] == s]
        min_data_sets = min(int(len(server_df)/2), 300)
        data_draw = server_df.iloc[min_data_sets-50:min_data_sets+50, :]

        plt.figure(figsize=(10, 5))
        plt.step(data_draw['task_id'], data_draw['service'], where='mid', linewidth=2)

        plt.xlabel('Task ID')
        plt.ylabel('Service')

        yticks = sorted(data_draw['service'].unique())
        ylabels = [service_mapping.get(y, str(y)) for y in yticks]
        plt.yticks(yticks, ylabels)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        title_ = title.replace("/", f"/queue_sequence_server_{s}_").replace(".csv",".png")
        histogram(data_draw,'service', 'queueing task type', title)
        save_path_ = os.path.join(save_path, title_.replace(' ', '_'))
        plt.savefig(save_path_, dpi=300)
        plt.close()
def analyasis_file(dir_path,f):
    file = pd.read_csv(os.path.join(dir_path, f))
    file = add_server_service(file)
    file = file.sort_values(by ='task_id')
    
    save_path_file = os.path.join(dir_path.split('/')[-1], f)
    os.makedirs(os.path.join(save_path, dir_path.split('/')[-1]), exist_ok=True)
    print(save_path_file)
    draw_server_service_distribution(file, save_path_file)
    queueing_time_distribution(file, save_path_file)
    histogram(file,'total_delay', 'total_delay', save_path_file)
    file.to_csv(os.path.join(save_path, dir_path.split('/')[-1],f"add_queue_server_infor_{f}"), index=False)
    
    
def analysis(dir_path):
    full_dir_path = os.path.join(raw_path, dir_path)
    print(full_dir_path)

    files = [f for f in os.listdir(full_dir_path) 
            if os.path.isfile(os.path.join(full_dir_path, f)) and f.lower().endswith(".csv")]
    [analyasis_file(full_dir_path, f) for f in files]

for d in directories:
    analysis(d)