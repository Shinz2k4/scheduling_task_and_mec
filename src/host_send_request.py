import asyncio, argparse
import httpx
import numpy as np
import os
import yaml
import json
import re, time
from utils.utils_func import get_active_service

async def send_tasks(task_num, url, request = "", docker = None, id = None, current_state_information = [], model = "None", id_picture = None, required_cpu = 3.1, deadline = 1.0,client = None):
    
    list_docker, list_service_in_docker = get_active_service()
    port_base = 10000
    
    cnt=0
    response = None
    while task_num:
        if id_picture is None:
            id_picture = np.random.randint(0, len(os.listdir("./../val2017")))
        with open("./../configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        docker = int(docker)
        if docker is None and len(list_docker) != 0:
            docker = np.random.choice(list_docker)
        elif docker is None and len(list_docker) == 0:
            docker = 1 
            model = "None"
        elif docker is not None and model == "None" and len(list_docker) != 0:
            model = np.random.choice(list_service_in_docker[docker])
            model = "None"
        port = int(port_base)+ int(docker)
        
        current_time = time.time()
        if id is None:
            id = cnt
        payload = {
            "task_id": id,
            "description": f"{request}",
            "inference": ["--id_picture", str(id_picture), "--model", model],
            "docker": str(docker),
            'port_base': str(port),
            "arrival_time": current_time,
            "end_time": 0.0,
            "current_state_information": str(current_state_information),
            "require_cpu": float(required_cpu ) if required_cpu is not None else 3.1,
            "deadline": float(deadline) if deadline is not None else 1.0
        }
        # print(f"Sending task with request {request} id_picture = {id_picture}, model={model}  --> {current_state_information}")
        
        if client is not None:
            response = await client.post(url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=1, http2=False) as client:
                response = await client.post(url, json=payload)  
                # print(response.json())
        task_num -= 1
        cnt += 1

    return response
if __name__ == "__main__":
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=str, required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--docker", type=str,default=None)
    # ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=str, default="10000")
    ap.add_argument("--id", type=int, default=0, required=False)
    ap.add_argument("--state", type=str, required=False, default=[])    
    ap.add_argument("--model", type=str, required=False)
    ap.add_argument("--id_picture", type=str, required=False)
    ap.add_argument("--required_cpu", type=float, required=False)
    ap.add_argument("--deadline", type=float, required=False)

    args = ap.parse_args()
    port = args.port
    asyncio.run(send_tasks(task_num=args.num, url=f"http://localhost:{port}/handle_host_request", 
                        request=args.request, docker = args.docker, id = args.id, current_state_information = args.state, model=args.model, id_picture=args.id_picture, required_cpu=args.required_cpu, deadline=args.deadline))
