from multiprocessing import queues
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import asyncio, httpx
from utils.utils_func import get_docker_metrics_by_name, thread_func
from host_send_request import send_tasks
from fastapi import FastAPI, Request
from collections import deque
import time
import threading
import socket, json

class OffloadingEnv(gym.Env):
    """
    Môi trường DRL mô phỏng việc offloading task tới nhiều server.
    Quan sát: trạng thái queue của từng server + khả năng xử lý.
    Hành động: chọn server để gửi task.
    """
    def __init__(self, num_servers=3, max_queue=10, port_base=10000, update_interval=0.1):
        super().__init__()
        self.num_servers = num_servers
        self.max_queue = max_queue
        self.update_interval = update_interval
        self.socket_listeners = []
        
        # Action và observation space
        self.action_space = spaces.Discrete(num_servers)
        obs_low = np.array([0]*num_servers + [0]*num_servers, dtype=np.float32)
        obs_high = np.array([max_queue]*num_servers + [1]*num_servers, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Server info
        self.server_host = ["0.0.0.0"]*num_servers
        self.server_port = [port_base + i for i in range(1, num_servers+1)]
        #socket_listener
        self.create_socket_listener()
        
        # State và cache Docker metrics
        self.state = None
        self._docker_metrics_cache = []
        self._docker_metrics_task = None
        self.all_queue_end = False
        self.use_history_task_observation = False
        self.length_task_history = 5
        self.varience = {}
        self.client = httpx.AsyncClient(http2=False)
        
    def create_socket_listener(self):
        self.close_socket()
        self.socket_listeners = []
        for i in range(len(self.server_port)):
            SOCKET_PATH = f"./../tmp/docker_sockets/container_{self.server_port[i]}.sock"
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(SOCKET_PATH)
            self.socket_listeners.append(client)
    
    def close_socket(self):
        if len(self.socket_listeners) == 0:
            return
        for socket in self.socket_listeners:
            socket.close()        
        self.socket_listeners = []

    def set_use_history_task_observation(self, value: bool):
        self.use_history_task_observation = value
        self.historical_tasks = [deque([-1] * self.length_task_history, maxlen=self.length_task_history) for _ in range(self.num_servers)]
    
    def reset_historical_tasks(self):
        if self.use_history_task_observation:
            self.historical_tasks = [deque([-1] * self.length_task_history, maxlen=self.length_task_history) for _ in range(self.num_servers)]
        
    def update_historical_tasks(self, server_idx, task_idx):
        if self.use_history_task_observation:
            self.historical_tasks[server_idx].append(task_idx)
    
    async def ainit(self):
        """Khởi tạo async: chạy background task cập nhật Docker metrics"""
        if self._docker_metrics_task is None or self._docker_metrics_task.done():
            self._docker_metrics_task = asyncio.create_task(
                self._update_docker_metrics_periodically()
            )
            print("[Env] Docker metrics task started.", flush=True)
        else:
            print("[Env] Docker metrics task already running.", flush=True)
        return self
    async def _update_docker_metrics_periodically(self):
        while True:
            try:
                result_container = {}
                # Chạy thread blocking an toàn
                await asyncio.to_thread(thread_func, result_container)
                self._docker_metrics_cache = result_container['metrics']
            except Exception as e:
                print(f"[Docker Metrics Error] {e}", flush=True)
            await asyncio.sleep(self.update_interval)
    
    async def fetch_queue(self, idx, port, request = "state"): 
        # url = f"http://localhost:{port}/handle_host_request"
        # async with httpx.AsyncClient(timeout=1, http2=True) as client:
        #     r = await send_tasks(task_num=1, url=url, request="state", docker=idx+1, client=client)
        #     data = r.json() 
        #     queue_list = data.get("queue", [])
        # self.create_socket_listener()
        SOCKET_PATH = f"./../tmp/docker_sockets/container_{port}.sock"
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        try:
            message = json.dumps({"request": request}).encode()
            await asyncio.to_thread(client.sendall, message)
            response = await asyncio.to_thread(client.recv, 165536) 
            data = json.loads(response.decode())
            if request!= 'state':
                return data
            else:
                return data.get("queue", [])
        except (ConnectionError, json.JSONDecodeError) as e:
            print(f"Error fetching from client {idx}: {e}")
            return []
        
    async def get_observation(self):
        """Lấy state hiện tại: Docker metrics + queue lengths"""        
        queues = [self.fetch_queue(i,self.server_port[i]) for i in range(self.num_servers)] 
        queues = await asyncio.gather(*queues)
        # print(self._docker_metrics_cache)
        server_information_ram_cpu = self._docker_metrics_cache.copy()
        if not server_information_ram_cpu:
            server_information_ram_cpu = np.random.uniform(0,1,size=self.num_servers*6).tolist()

        self.state = []
        inter = int(len(server_information_ram_cpu)/self.num_servers)
        self.all_queue_end = True
        for i in range(self.num_servers):
            start = i*inter
            end = start + inter
            self.state += server_information_ram_cpu[start:end]
            self.state.append(len(queues[i]))
            print(f"queue length ----------- {int(i)}",len(queues[int(i)]))
            if len(queues[i])>0:
                self.all_queue_end = False
            else:
                self.varience = {}
            if self.use_history_task_observation:
                self.state += list(self.historical_tasks[i])
        return self.state
    def is_all_queue_end(self):
        return self.all_queue_end
    async def get_reward(self):
        queues = [self.fetch_queue(i, self.server_port[i],request="result")
                for i in range(self.num_servers)]
        queues = await asyncio.gather(*queues)
        merged = {}
        try:
            for q in queues:
                merged.update(q['results'])
        except Exception as e:
            print(f"an erroes {e}")
        return merged
    async def get_queues(self):
        """Lấy trạng thái queue hiện tại của các server"""        
        queues = [self.fetch_queue(i,self.server_port[i]) for i in range(self.num_servers)] 
        queues = await asyncio.gather(*queues)
        return queues
    async def get_reward_by_queing_varience(self, taks_id, server_id):
        queues = [self.fetch_queue(i, self.server_port[i], request="state") for i in range(self.num_servers)] 
        queues = await asyncio.gather(*queues)
        reward = {}
        for idx, q in enumerate(queues):
            if server_id == idx:
                if idx in self.varience.keys():
                    pre_data = self.varience[idx]
                    n = pre_data['n']
                    mean_old = pre_data['mean']
                    M2 = pre_data['M2']

                    n += 1
                    mean = mean_old + (taks_id-mean_old)/n
                    M2 =  (M2*(n-1) + (taks_id - mean_old)*(taks_id - mean))/n

                    self.varience[idx] = {
                        'M2': M2,
                        'n': n,
                        'mean': mean
                    }
                    reward[idx] = M2*0.5 + 0.5*len(queues[idx])
                else:
                    self.varience[idx] = {
                        'M2': 0.0,
                        'n': 1,
                        'mean': taks_id
                    }
                    reward[idx] = 0.0*0.5 + 0.5*len(queues[idx])
            else:
                continue
        return reward
    
    def reset(self):
        self.state = np.concatenate([
            np.random.randint(0, self.max_queue//2, size=self.num_servers),
            np.random.rand(self.num_servers)  # khả năng xử lý (0-1)
        ]).astype(np.float32)
        return self.state
    
    def step(self, action):
        queue_lengths = self.state[:self.num_servers]
        capacities = self.state[self.num_servers:]
        reward = -queue_lengths[action] + capacities[action]
        
        queue_lengths[action] += 1
        
        queue_lengths = np.maximum(queue_lengths - capacities, 0)
        
        self.state = np.concatenate([queue_lengths, capacities]).astype(np.float32)
        
        done = False  
        info = {}
        return self.state, reward, done, info
    
    def render(self):
        queue_lengths = self.state[:self.num_servers]
        capacities = self.state[self.num_servers:]
        print("Queues:", queue_lengths, "Capacities:", capacities)
        
async def main_loop(env):
    await env.ainit()  # chạy async init, tạo background task
    while True:
        result = await env.get_observation()
        print("Observation:", result)
        await asyncio.sleep(1)  # tránh busy loop

if __name__ == "__main__":
    env = OffloadingEnv(num_servers=3)
    asyncio.run(main_loop(env))
    
    
