# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess, sys
from .inference import inference
from .launch import lauch_service
import json
import os
import asyncio, httpx
app = FastAPI()
import time
import orjson
from fastapi.responses import JSONResponse, Response
from collections import deque
import stat
from sort_queue_deadline_algorithm import sort_queue_deadline

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
    
       

def default(obj):
    if hasattr(obj, "dict"):       # Pydantic model
        return obj.dict()
    elif hasattr(obj, "__dict__"):  # Plain Python class
        return obj.__dict__
    elif isinstance(obj, deque):    # Convert deque to list for JSON
        return list(obj)
    raise TypeError

run_dict = {"install": [sys.executable, "launch.py"],
            "inference" : [sys.executable, "inference.py"],
            "kill":[sys.executable, "kill_all.py"]
            }

queue = asyncio.Queue()
results = asyncio.Queue()
compute_done = True

SERVICE_PATH = "./../../service"

worker_paused = asyncio.Event()
worker_paused.set()


async def worker():
    global compute_done
    while True:
        wait_time = time.perf_counter()
        if not queue.empty() and compute_done:
            task: Task = await queue.get()
            await worker_paused.wait()
            total_wait = time.perf_counter() - wait_time
            print(f"--------------> total_wait because of lock {total_wait}")
            worker_paused.clear()
            compute_done = False
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
                    url = "http://host.docker.internal:15000/catch_results"
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(url, json=payload)  # chỉ dùng json=payload
                        print(r.status_code, r.text)
            except Exception as e:
                await results.put([task.task_id, "error", str(e)])
            finally:
                queue.task_done()
                worker_paused.set()
            compute_done = True
        await asyncio.sleep(0.0000000000000000000001)

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
            t1 = time.perf_counter()
            j = orjson.dumps({"queue": queue._queue}, default=default)
            t2 = time.perf_counter()
            print(f"handling state event {t2-t1}")
            return Response(
                content=j,
                media_type="application/json"
            )
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


#===============================================
# WEBSOCKET HANDLEING
#===============================================

port = int(os.environ.get("CONTAINER_PORT", 10000))
print(port)
SOCKET_PATH = f"./../../tmp/docker_sockets/container_{port}.sock"

async def socket_server():
    global queue, results
    async def handle_client(reader, writer):
        data = await reader.read(165536)
        if not data:
            return
        task = json.loads(data.decode())
        print(f"[Socket-{port}] Received task: {task}")
        if task['request'] == 'state':
            result = orjson.dumps({"queue": queue._queue}, default=default)
        elif task['request'] == 'result':
            cur_result = {}
            worker_paused.clear()
            while not results.empty():
                try:
                    re: dict = await results.get()
                    cur_result.update(re)
                except Exception as e:
                    print (f"Error while fetching result: {e} for result {re}")
            worker_paused.set()
            result = json.dumps({"results": cur_result}, default=default).encode('utf-8')
            print(result)
        writer.write(result)
        await writer.drain()
    try:
        server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
        os.chmod(SOCKET_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print(f"[Socket-{port}] Listening on {SOCKET_PATH}")
        async with server:
            await server.serve_forever() 

    except asyncio.CancelledError:
        print(f"[Socket-{port}] Listener was deliberately cancelled.")
    except Exception as e:
        print(f"[Socket-{port}] CRITICAL ERROR in server loop: {e}")
    finally:
        if os.path.exists(SOCKET_PATH):
             os.remove(SOCKET_PATH)
        print(f"[Socket-{port}] Listener shutdown complete.")

SOCKET_TASK_REF = None 
@app.on_event("startup")
async def startup_all_tasks():
    global SOCKET_TASK_REF
    asyncio.create_task(worker()) 
    SOCKET_TASK_REF = asyncio.create_task(socket_server())
    print("All background services (Worker & UDS Listener) started successfully.")
@app.on_event("shutdown")
async def shutdown_all_tasks():
    global SOCKET_TASK_REF
    if SOCKET_TASK_REF:
        SOCKET_TASK_REF.cancel()
        try:
            await SOCKET_TASK_REF
        except asyncio.CancelledError:
            pass
        print("UDS Listener background task cancelled gracefully.")
