import os, io, time, argparse
from fastapi import FastAPI, File, UploadFile, Header, Request, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import uvicorn
from typing import Dict, Any
import logging, threading
from pydantic import BaseModel
import pandas as pd
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

class RestartPayload(BaseModel):
    name: str
torch.set_num_threads(1)
saver_thread = None
stop_event = threading.Event()
class Task(BaseModel):
    task_id: int
    description: str
    inference: list = None
    docker: int
    port_base: int
app = FastAPI()

df = {"task_id": [],
      
    "arrival_time": [],
    "end_time": [],
    "total_delay": [],
    "id_picture" : [],
    "current_state_information" : [],
    "description": [],
    "compute_delay": [], #one task without waiting time
    "results": []
    }


def save_periodically(interval, file_name):
    while not stop_event.is_set():
        if df["task_id"]: 
            pd.DataFrame(df).to_csv(file_name, index=False)
            print(f"[Saver] Saved to {file_name}")
        stop_event.wait(interval)  # dừng sớm nếu stop_event được set

def start_saver(interval, file_name):
    t = threading.Thread(target=save_periodically, args=(interval, file_name), daemon=True)
    t.start()
    return t

@app.post("/restart_saver_no_reset_df")
async def restart_saver(payload: RestartPayload):
    global saver_thread, stop_event
    if saver_thread and saver_thread.is_alive():
        stop_event.set()
        saver_thread.join()

    stop_event = threading.Event()
    saver_thread = start_saver(10, payload.name)
    print(f"reset saver without reset df already done!!!")
    return {"status": f"saver thread restarted: , file: {payload.name}"}

@app.post("/restart_saver")
async def restart_saver(payload: RestartPayload):
    global saver_thread, stop_event, df
    df = {"task_id": [],
    "arrival_time": [],
    "end_time": [],
    "total_delay": [],
    "id_picture" : [],
    "current_state_information" : [],
    "description": [],
    "compute_delay": [], #one task without waiting time
    "results": [],
    "deadline": [],
    "required_cpu": []
    }
    if saver_thread and saver_thread.is_alive():
        stop_event.set()
        saver_thread.join()

    stop_event = threading.Event()
    saver_thread = start_saver(10, payload.name)
    print(f"reset saver and reset df already done!!!")
    return {"status": f"saver thread restarted: , file: {payload.name}"}

@app.post("/catch_results")
async def infer(payload: dict):
    task = payload["task"]
    compute_delay = float(payload["compute_delay"])
    result = payload["result"]
    df["task_id"].append(task["task_id"])
    df["arrival_time"].append(task["arrival_time"])
    df["end_time"].append(task["end_time"])
    df["total_delay"].append(task["total_delay"])
    df["description"].append(task["description"])
    df["compute_delay"].append(compute_delay)
    df["results"].append(result)
    df["current_state_information"].append(task['current_state_information'])    
    df["id_picture"].append(task['id_picture'])
    df["deadline"].append(task.get('deadline', None))
    df["required_cpu"].append(task.get('require_cpu', None))

    print(f"Received task {task['task_id']}: delay={task['total_delay']}, result={result}", flush=True)
    return JSONResponse("done")

@app.get("/health")
async def health(MODEL_NAME):
    return {"model": MODEL_NAME, "status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 15000)))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
    # uvicorn.run(app, host=args.host, port=args.port, log_level="debug")