import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import ast
import re

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
    #if drl training then only draw after training has been done half of the data
    if path.find("drl_train")!=-1:
        min_data_sets = int(len(df)/2)
        df = df.iloc[min_data_sets:, :]
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
    return stats
    
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
    file.to_csv(os.path.join(save_path, dir_path.split('/')[-1],f"add_queue_server_infor_{f}"), index=False)
    return f, histogram(file,'total_delay', 'total_delay', save_path_file), file

drl_train = {'data': [], 'num': []}
drl_train_history = {'data': [], 'num': []}
drl_eval = {'data': [], 'num': []}
drl_eval_task_history = {'data': [], 'num': []}
processing_estimation = {'data': [], 'num': []}
random = {'data': [], 'num': []}
added = []
def add_data(value, target, idx):
    if idx  <= len(target['data'])-1:
        value = (value + target['data'][idx]*target['num'][idx])/(target['num'][idx]+1)
        target['num'][idx] += 1
        target['data'][idx] = value
    else:
        target['data'].append(value)
        target['num'].append(1)
def analysis(dir_path):
    full_dir_path = os.path.join(raw_path, dir_path)
    print(full_dir_path)
    match = re.match(r"output_file_(\d+)_service_(\d+)_users", dir_path)
    files = [f for f in os.listdir(full_dir_path) 
            if os.path.isfile(os.path.join(full_dir_path, f)) and f.lower().endswith(".csv")]
    files = sorted(files)
    longest_file = None
    max_lh = 0
    added.append(match[2])
    for idx, f in enumerate(files):
        _, stats, modified_files= analyasis_file(full_dir_path, f)
        if len(modified_files)>max_lh:
            max_lh = len(modified_files)
            longest_file = modified_files
        if match is not None and f.find("drl_train_with_history_task_observation")!=-1:
            add_data(stats['mean'], drl_train_history, len(added)-1)
        elif match is not None and f.find("drl_train_seed")!=-1:
            add_data(stats['mean'], drl_train,len(added)-1)
        elif match is not None and f.find("drl_prediction_seed")!=-1:
            add_data(stats['mean'], drl_eval,len(added)-1)
        elif match is not None and f.find("drl_prediction_with")!=-1:
            add_data(stats['mean'], drl_eval_task_history,len(added)-1)
        elif match is not None and f.find("esimated_processing_time")!=-1:
            add_data(stats['mean'], processing_estimation,len(added)-1)
        elif match is not None and f.find("random")!=-1:
            add_data(stats['mean'], random,len(added)-1)
            
        
    if longest_file is not None:
        # Get unique task types
        all_tasks = longest_file['service'].unique()
        
        model_map = {
            3: "resnet34",
            9: "ssd"
        }

        data = []
        labels = []

        for t in all_tasks:
            model_name = model_map.get(t, "mobilenet_v2")
            task_df = longest_file[longest_file['service'] == t]
            # Collect cost values (assuming column 'cost' exists)
            if 'compute_delay' in task_df.columns:
                data.append(task_df['compute_delay'])
                labels.append(model_name)

        # Plot combined boxplot
        plt.figure(figsize=(12, 6))
        box = plt.boxplot(data, labels=labels, patch_artist=True)

        # Apply colors
        colors = plt.cm.Set3.colors
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title('Compute Delay  Distribution by Model', fontsize=14, weight='bold')
        plt.ylabel('Compute Delay', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=15)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(save_path, 'compute_delay_distribution.png'))
        plt.close()


n_servers = [3]
for n in n_servers:
    for d in directories:
        match = re.match(r"output_file_(\d+)(?:_service)?_(\d+)_users", d)
        if match is not None and int(match[1])==n:
            analysis(d)
        else:
            print(f"skip {d} not match {n} servers")
    print("drl_train", drl_train)
    print("drl_eval", drl_eval)
    print("drl_eval_task_history", drl_eval_task_history)
    print("processing_estimation", processing_estimation)
    print("random", random)
    print("drl_train_history", drl_train_history)
    added = [int(val) for val in added]
    sorted_idx = sorted(range(len(added)), key=lambda i: added[i])
    
    drl_train_vals = [drl_train['data'][u] for u in sorted_idx]
    drl_vals = [drl_eval['data'][u] for u in sorted_idx]
    drl_eval_task_history_vals = [drl_eval_task_history['data'][u] for u in sorted_idx]
    processing_vals = [processing_estimation['data'][u] for u in sorted_idx]
    random_vals = [random['data'][u] for u in sorted_idx]
    drl_train_history_values = [drl_train_history['data'][u] for u in sorted_idx]
    plt.figure(figsize=(10, 5))
    users = [added[u] for u in sorted_idx]
    plt.plot(users, drl_train_vals, marker='o', label='DRL Train')
    plt.plot(users, drl_train_history_values, marker='.', label='DRL Train History Task')
    plt.plot(users, drl_vals, marker='s', label='DRL eval')
    plt.plot(users, drl_eval_task_history_vals, marker='v', label='DRL eval History Task')
    plt.plot(users, processing_vals, marker='^', label='Processing Estimation')
    plt.plot(users, random_vals, marker='x', label='Random')

    plt.xlabel('Number of Users')
    plt.ylabel('Performance Metric')
    plt.title('Performance Comparison by Number of Users')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'performance_comparison_{n}_servers.png'), dpi=300)
    plt.close()
    drl_train = {}
    drl_train_history = {}
    drl_eval = {}
    drl_eval_task_history = {}
    processing_estimation = {}
    random = {}