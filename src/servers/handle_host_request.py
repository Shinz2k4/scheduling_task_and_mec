# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess, sys
from .inference import inference
from .launch import lauch_service
import json
import asyncio, httpx
app = FastAPI()
import time
from fastapi.responses import JSONResponse


class Task(BaseModel):
    task_id: int
    description: str
    inference: list = None
    docker: int
    port_base: int
    arrival_time: float
    end_time: float
    current_state_information: str
    require_cpu: float 
    deadline: float 
    
    def __hash__(self):
        inference_tuple = tuple(self.inference) if self.inference is not None else ()        
        current_state_information = tuple(self.current_state_information) if self.current_state_information is not None else ()

        return hash((
            self.task_id,
            self.description,
            inference_tuple,
            self.docker,
            self.port_base,
            self.arrival_time,
            self.end_time,
            current_state_information,
            self.require_cpu,
            self.deadline
        ))
    

run_dict = {"install": [sys.executable, "launch.py"],
            "inference" : [sys.executable, "inference.py"],
            "kill":[sys.executable, "kill_all.py"]
            }

queue = asyncio.Queue()
results = asyncio.Queue()


SERVICE_PATH = "./../../service"

worker_paused = asyncio.Event()
worker_paused.set()

def sort_queue_deadline(queue: asyncio.Queue):
    """
    sắp xếp lại queue theo algorithm 1 
    """
    print("Sorting queue ...")
    if queue.empty():
        return []
    
    tasks = list(queue._queue)
    N = len(tasks)
    queue_sorted = [0] * N
    
    tasks = sorted(tasks, key=lambda t: t.deadline)
    to_remove = set()
    for i, task in enumerate(tasks):
        index = next(
            (j for j, t in enumerate(tasks) if t.require_cpu == task.deadline),
            None
        )

        if index is None:
            continue

        queue_sorted[index] = task.require_cpu
        outage = False
    
        for j in range(index, N):
            if queue_sorted[j] != 0:
                
                required_cpu = getattr(task, "require_cpu")
                deadline = getattr(task, "deadline")

                total = sum(required_cpu * queue_sorted[k] / (3.1*10**9) for k in range(j + 1))
                if total > deadline * queue_sorted[j]:
                    outage = True
                    break
        
        if outage:
            queue_sorted[index] = 0
            to_remove.add(task)
    remaining = [t for t in tasks if t not in to_remove]

    queue._queue.clear()
    for t in remaining:
        queue.put_nowait(t)
                
    return queue_sorted


async def worker():
    while True:
        wait_time = time.perf_counter()
        if not queue.empty():
            task: Task = await queue.get()
            await worker_paused.wait()
            total_wait = time.perf_counter() - wait_time
            print(f"--------------> total_wait because of lock {total_wait}")
            worker_paused.clear()
            
            try:
                id_picture = int(task.inference[1])
                model = task.inference[3] 
                dt, result = await inference(id_picture, model, task.port_base)
                current_time = time.time() - total_wait
                print(f"[Worker] Finished inference request for task {task.task_id}")
                
                total_delay = current_time - float(task.arrival_time)
                await results.put({task.task_id: (dt, result, total_delay)})
                print(f"======================\ntotal_delay {total_delay}\n======================")
                
                payload = {
                        "task": {
                            "task_id": task.task_id,
                            "description": task.description,
                            "id_picture" : task.inference[1],
                            "arrival_time": task.arrival_time,
                            "end_time": str(current_time),
                            "total_delay": str(total_delay),
                            "current_state_information": str(task.current_state_information),
                            "require_cpu": str(task.require_cpu),
                            "deadline": str(task.deadline)
                        },
                        "compute_delay": dt,
                        "result": result  
                    }
                try:
                    if sys.platform.startswith("darwin"):
                        url = "http://host.docker.internal:15000/catch_results"
                    elif sys.platform.startswith("linux"):
                        url = "http://0.0.0.0:15000/catch_results"
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(url, json=payload)  # chỉ dùng json=payload
                        print(r.status_code, r.text)
                except:
                    pass
            except Exception as e:
                await results.put([task.task_id, "error", str(e)])
            finally:
                queue.task_done()
                worker_paused.set()
                
        await asyncio.sleep(0.5)

@app.post("/handle_host_request")
async def handle_host_request(task: Task):
    global queue, results
    
    print(f"Received task: {task.task_id}, description: {task.description}, port {task.port_base}, require_cpu: {task.require_cpu}, deadline: {task.deadline}")
    try:
        if task.description == "install":
            list_active_service = lauch_service(task.port_base)
            filename = f"{SERVICE_PATH}/active_services_in_docker_{int(task.docker)}-th.json"
            with open(filename, "w") as f:
                json.dump(list_active_service, f, indent=2)
        elif task.description == "inference":
            
            # fdo
            print(f"Queue size before adding task: {queue.qsize()}")
            await queue.put(task)
            sort_queue_deadline(queue)
            print(f"Queue has been sorted.")
            print(f"Queue size after adding task: {queue.qsize()}")
            return {"status": "queued", "task_id": task.task_id,"current result": []}
        elif task.description == "kill":
            proc = subprocess.Popen(run_dict[task.description ])
        elif task.description == "state":
            cur_queue = []
            for q in list(queue._queue):
                cur_queue.append(q.dict())
            return JSONResponse({
                "queue": cur_queue
            })
        elif task.description == "result":
            cur_result = {}
            worker_paused.clear()
            while not results.empty():
                re: dict = await results.get()
                cur_result.update(re)   # merge dict re vào cur_result
            worker_paused.set()
            return JSONResponse({
                "results": cur_result
            })
        elif task.description == "clearq":
            worker_paused.clear()
            queue = asyncio.Queue()
            results = asyncio.Queue()
            worker_paused.set()
            
            return JSONResponse({
                "results": "clear queue and results success"
            })
        elif task.description == "ping":
            return {"status": "pong", "task_id": task.task_id}
        elif task.description == "train":   
            pass
        return {"status": "success", "task_id": task.task_id}

    except Exception as e:
        print(f"Error while processing task: {e}")
        return {"status": "error", "message": str(e)}
    
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())
