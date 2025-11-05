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

import threading

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
        
        # Action và observation space
        self.action_space = spaces.Discrete(num_servers)
        obs_low = np.array([0]*num_servers + [0]*num_servers, dtype=np.float32)
        obs_high = np.array([max_queue]*num_servers + [1]*num_servers, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Server info
        self.server_host = ["0.0.0.0"]*num_servers
        self.server_port = [port_base + i for i in range(1, num_servers+1)]
        
        # State và cache Docker metrics
        self.state = None
        self._docker_metrics_cache = []
        self._docker_metrics_task = None
        self.all_queue_end = False
        self.use_history_task_observation = False
        self.length_task_history = 5
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

    async def get_observation(self):
        """Lấy state hiện tại: Docker metrics + queue lengths"""        
        async def fetch_queue(idx, host, port): 
            url = f"http://localhost:{port}/handle_host_request" 
            r = await send_tasks(task_num=1, url=url, request="state", docker=idx+1) 
            data = r.json() 
            queue_list = data.get("queue", [])
            return queue_list 
        queues = [fetch_queue(i, self.server_host[i], self.server_port[i]) for i in range(self.num_servers)] 
        queues = await asyncio.gather(*queues)
        print(f"Queues fetched : {queues}")

        # for i in range(self.num_servers):
        #     if fast_detect_outage(queues[i], new_task, current_clock=env.current_time):
                
        #         queues[i].append(new_task)
        #         assigned_server = i
        #         break
        # else:
        #     print(f"[!] Task {new_task.id} bị outage — không server nào lập lịch được.")

        
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
            print(f"queue length ----------- {int(i)}", len(queues[int(i)]))
            if len(queues[i]) > 0:
                self.all_queue_end = False
            if self.use_history_task_observation:
                self.state += list(self.historical_tasks[i])
        
        return self.state
    async def get_queues(self):
        """Lấy trạng thái queue hiện tại của các server"""        
        async def fetch_queue(idx, host, port): 
            url = f"http://localhost:{port}/handle_host_request" 
            r = await send_tasks(task_num=1, url=url, request="state", docker=idx+1) 
            data = r.json() 
            queue_list = data.get("queue", [])
            return queue_list 
        queues = [fetch_queue(i, self.server_host[i], self.server_port[i]) for i in range(self.num_servers)] 
        queues = await asyncio.gather(*queues)
        return queues
    
    def is_all_queue_end(self):
        return self.all_queue_end
    async def get_reward(self):
        async def fetch_queue(idx, host, port):
            url = f"http://localhost:{port}/handle_host_request"
            r = await send_tasks(task_num=1, url=url, request="result", docker=idx+1)
            data = r.json()
            return data
        queues = [fetch_queue(i, self.server_host[i], self.server_port[i])
                for i in range(self.num_servers)]
        queues = await asyncio.gather(*queues)
        merged = {}
        # print(queues)
        for q in queues:
            merged.update(q['results'])
        return merged
        
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
    
    
