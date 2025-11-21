import subprocess, sys, json, time, yaml
import os
import time
import numpy as np
import platform
import psutil
import threading


def lauch_service(port_base:int = 10000):
    # os.system("clear")
    with open("./../configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    pids = {"services": [], "scheduler": None}

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    str_lst = 100*(int(str(port_base)[-1])+1)
    print(f"------ str_lst {str_lst}")
    
    sch_port = int(cfg.get("scheduler", {}).get("port", 8000))
    sch_host = str(cfg.get("scheduler", {}).get("host", "0.0.0.0"))
    sch_port = port_base + sch_port +str_lst
    proc = subprocess.Popen([sys.executable, "./servers/scheduler_proc.py", "--host", sch_host, "--port", str(sch_port)])
    pids["scheduler"] = proc.pid
    print(f"Scheduler started (pid={proc.pid})")
    pros = []

    keys = list(cfg.get("services", {}).keys())

    #because if index of serivce high then it will require more time for inference
    #so we will set it with low probability for random choice
    prob = np.array([1/(i+1) for i in range(len(keys))])
    prob = prob/prob.sum()

    # list_service = {}
    # for i in range(cfg.get("server", {}).get("max_service",1)):
    #     model = np.random.choice(range(len(keys)), p=prob)
    #     if (i <=2) and model>6:
    #         list_service[model] = cfg.get("services", {}).get(keys[model], [])
    #         break
    #     if i>2 and model>6:
    #         continue
    #     list_service[keys[model]] = cfg.get("services", {}).get(keys[model], [])
    list_service = cfg.get("services", {})
    print("Services to launch:", list_service)

    for model, endpoints in list_service.items():
        for ep in endpoints:
            host, port = ep.split(":")
            port = int(port_base)+ int(port)+ str_lst
            print(f"current port -------- {port_base} new port {port}")
            
            cmd = [sys.executable, "./servers/infer_service_proc.py", "--model", model, "--host", host, "--port", str(port)]
            with open(f"{log_dir}/{model}_log.txt", "w") as log_file:
                proc = subprocess.Popen(cmd, stderr=log_file)
                pros.append(proc)
            pids["services"].append(proc.pid)
            print(f"Started {model} at {host}:{port} (pid={proc.pid})")
            time.sleep(0.2)  # stagger


    with open("pids.json", "w") as f:
        json.dump(pids, f, indent=2)
    print("PIDs saved to pids.json. Dừng bằng: python kill_all.py")
    
    return list_service