import asyncio, argparse
import httpx
import numpy as np
import re, sys, subprocess
from env import OffloadingEnv
from ddqn_agent import DDQNAgent
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import time
from utils.utils_func import get_active_service, get_feature
import os
import torch
from trainer_processing_time import DelayPredictor
from sklearn.preprocessing import StandardScaler
import joblib
from host_send_request import send_tasks

import threading

import pandas as pd
N_SERVER = 4

df = {"id": [], "id_picture":[], "predict_cost": [] }
experiment_types = {
    0: 'random',
    1: 'drl_train',
    2: 'drl_prediction',
    3: 'esimated_processing_time',
    4: 'drl_train_with_history_task_observation',
    5: 'drl_prediction_with_history_task_observation'
}

async def run(n_users=10, lamd=1.1, port_base=10000, docker_min_max=[], duration=1000, output_file = "results", N_SERVER = 4, experiment_type = 0, LGOBAL_SEED = 45):
    list_docker, list_service_in_docker = get_active_service()
    output_file = "./../"+output_file + f"_{len(list_service_in_docker[1])}_service_{n_users}_users"
    print("output_file", output_file)
    os.makedirs(output_file, exist_ok=True)
    payload = {"name": f"{output_file}/results_n_server_{N_SERVER}_n_user_{n_users}_{experiment_types[experiment_type]}_seed_{LGOBAL_SEED}.csv"}
    url = "http://127.0.0.1:15000/restart_saver"
    drl_checkpoint = f'drl_servers_{N_SERVER}_service_{len(list_service_in_docker[1])}_n_users_{n_users}.pth'
    try:
        r = httpx.post(url, json=payload, timeout=30)
        print(r.status_code, r.text)
    except Exception as e:
        print(f"cannot save file {e} existing...")
        exit(0)
    rng = np.random.default_rng(LGOBAL_SEED)
    system_arrival_rate = n_users * lamd
    system_inter_arrival_rate = 1 / system_arrival_rate
    cnt = 0
    # vì OffloadingEnv có async nên ta tạo 1 loop riêng trong thread
    env = OffloadingEnv(num_servers=N_SERVER)
    if experiment_types[experiment_type].find('history_task_observation')!=-1:
        drl_checkpoint = drl_checkpoint.split('.')[0]+ "_with_history.pth"
        env.set_use_history_task_observation(True)
    await env.ainit()
    obs = await env.get_observation()
    queue = {}
    rewards = []
    all_reward = {}
    all_varience_reward = []
    agent = DDQNAgent(len(obs)+1, N_SERVER, batch_size = 512)
    if experiment_types[experiment_type].find('drl_prediction')!=-1:
        agent.load(drl_checkpoint)
    done = False
    save_dir = "train_result"
    fearture_vecs = get_feature(obs, id_picture=0, model=0, docker=1) #just for get size
    if experiment_types[experiment_type] == 'esimated_processing_time':
        load_model_estimate_processing_time = DelayPredictor(input_dim=len(fearture_vecs))
        load_model_estimate_processing_time.load_model(f"{save_dir}/pretrained_processing_estimation.pth")
    check_done  = 0
    if experiment_types[experiment_type] == 'esimated_processing_time':
        scaler = joblib.load(f"{save_dir}/scaler.pkl")
    client = httpx.AsyncClient(http2=False)

    while duration > 0:
        event = rng.exponential(system_inter_arrival_rate)
        print(event)
        await asyncio.sleep(event)
        id_picture = rng.integers(0, len(os.listdir("./../val2017")))
        model = rng.choice(list_service_in_docker[1])
        model_id = 0
        start_time = time.perf_counter()
        if model =="ssd":
            model_id = 9
        elif model =="resnet34":
            model_id = 3
        if experiment_types[experiment_type] == 'random':
            slected_docker = rng.integers(docker_min_max[0], docker_min_max[1])
            predict_cost = time.perf_counter() - start_time
        elif experiment_types[experiment_type].find('drl')!=-1:
            obs_pre = obs.copy()
            obs_pre.insert(0, model_id)
            slected_docker = agent.act(obs_pre) + 1
            predict_cost = time.perf_counter() - start_time
        elif experiment_types[experiment_type] == 'esimated_processing_time':
            def select_server(obs):
                slected_docker = 0            
                min_processing_predicted_time = float('inf')
                for i in range(N_SERVER):
                    fearture_vecs = np.array(get_feature(obs, id_picture, model_id, i+1)) #id docker from 1
                    fearture_vecs = np.reshape(fearture_vecs, (1, -1))
                    fearture_vecs=scaler.transform(fearture_vecs)
                    fearture_vecs = torch.from_numpy(fearture_vecs).float()
                    processing_predicted_time = load_model_estimate_processing_time(fearture_vecs)
                    if min_processing_predicted_time > processing_predicted_time:
                        slected_docker = i+1
                        min_processing_predicted_time = processing_predicted_time
                return slected_docker
            predict_cost = time.perf_counter() - start_time
    
            slected_docker = select_server(obs)
        t1 = time.perf_counter()
        df["id"].append(cnt)
        df["id_picture"].append(id_picture)
        df['predict_cost'].append(predict_cost)
        slected_port = port_base + slected_docker
        # kwargs_ = {
        #     'task_num': 1,
        #     'url': f"http://localhost:{slected_port}/handle_host_request",
        #     'request' : 'inference',
        #     'docker': slected_docker,
        #     'id': cnt,
        #     'model': model,
        #     'id_picture': id_picture,
        #     'client': client
        # }
        # t = threading.Thread(target=send_tasks, kwargs=kwargs_)
        # env.update_historical_tasks(slected_docker - 1, model_id)

        cmd = [
            sys.executable,
            "host_send_request.py",
            "--request", "inference",
            "--num", "1",
            "--docker", str(slected_docker),
            "--port", str(slected_port),
            "--id", str(cnt),
            "--state", f"{obs}", #save to train predict processing time
            "--model", str(model),
            "--id_picture", str(id_picture),
        ]
        env.update_historical_tasks(slected_docker - 1, model_id)
        subprocess.Popen(cmd)

        next_state = await env.get_observation()
        next_state_ = next_state.copy()
        next_state_.insert(0, model_id)
        obs.insert(0, model_id)
        queue[cnt] = [obs, slected_docker - 1, next_state_]
        reward = await env.get_reward()
        varience_reward = await env.get_reward_by_queing_varience(model_id, server_id = slected_docker -1)
        varience_rewards = -varience_reward[slected_docker-1]
        agent.remember(obs, slected_docker - 1, next_state_, varience_rewards, False)
        all_reward.update(reward)
        all_varience_reward.append(varience_rewards)
        obs = next_state

        #====================
        def process_rewards():
            nonlocal rewards, all_reward, queue, cnt, done
            del_r_key = []
            asyncio.run(env.get_observation())
            # print("env.is_all_queue_end() ---------------> ",env.is_all_queue_end())
            if done and env.is_all_queue_end():
                done= False
            for taskid in list(all_reward.keys()):
                try:
                    re_val = all_reward[taskid][2]
                except Exception as e:
                    continue
                try:
                    if re_val > 50 or (cnt > 0 and cnt%100 == 0 and experiment_types[experiment_type] == 'drl_train'):
                        env.varience = {}
                        done = True
                    rewards.append(-float(re_val))
                    del queue[int(taskid)]
                    del_r_key.append(taskid)  
                except Exception as e:
                    print(f"Chua tim thay {taskid} {e}")
                    continue
                
            for k in del_r_key:
                del all_reward[k]
            agent.update()

        await asyncio.to_thread(process_rewards)
        #====================
        loop_after_done = False
        while done:
            #sau khi done thì vẫn còn các nhiệm vụ trong queue, phải chờ cho chúng kết thúc rồi ms chuyển qua epoch mới
            await asyncio.to_thread(process_rewards)
            if not done:
                env.reset_historical_tasks()
            loop_after_done = True
        done = False
        if cnt > 0 and cnt % 100 == 0:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward')
            plt.savefig(f'./{output_file}/rewards_{experiment_types[experiment_type]}.png')
            plt.close()

            plt.plot(all_varience_reward)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward')
            plt.savefig(f'./{output_file}/rewards_varience_{experiment_types[experiment_type]}.png')
            plt.close()
            agent.save(drl_checkpoint)

            payload = {"name": f"{output_file}/results_n_server_{N_SERVER}_n_user_{n_users}_{experiment_types[experiment_type]}_seed_{LGOBAL_SEED}.csv"}

            url = "http://127.0.0.1:15000/restart_saver_no_reset_df"
            try:
                r = httpx.post(url, json=payload, timeout=30)
            except Exception as e:
                print(f"cannot save file {e} existing...")
                exit(0)
        check_done += 1

        duration -= event
        cnt += 1
        t2 = time.perf_counter()

        ''' 
        ==================================
        boi vi delay cua chuong trinh, chung ta buoc phai co 1 conpensate task
        thoi gian conpensate: t2-t1
        so  luong task conpensate = (t2-t1)/inter_arr_rate
        ==================================
        '''
        num_task_conpensate = int((t2-t1)/system_inter_arrival_rate)
        print(f"time counter {t2-t1}","---------- >>> ",num_task_conpensate, system_inter_arrival_rate)
        if not loop_after_done and num_task_conpensate:
            for i in range(num_task_conpensate):
                event = rng.exponential(system_inter_arrival_rate)
                await asyncio.sleep(event)
                id_picture = rng.integers(0, len(os.listdir("./../val2017")))
                model = rng.choice(list_service_in_docker[1])
                model_id = 0
                start_time = time.perf_counter()
                if model =="ssd":
                    model_id = 9
                elif model =="resnet34":
                    model_id = 3
                slected_docker = rng.integers(docker_min_max[0], docker_min_max[1])
                slected_port = port_base + slected_docker
                await env.get_reward_by_queing_varience(model_id, server_id = slected_docker -1)
                cmd = [
                    sys.executable,
                    "host_send_request.py",
                    "--request", "inference",
                    "--num", "1",
                    "--docker", str(slected_docker),
                    "--port", str(slected_port),
                    "--id", str(cnt),
                    "--state", f"{obs}", #save to train predict processing time
                    "--model", str(model),
                    "--id_picture", str(id_picture),
                ]
                queue[cnt] = "not record"
                env.update_historical_tasks(slected_docker - 1, model_id)
                subprocess.Popen(cmd)
                cnt += 1
        if num_task_conpensate:
            obs = await env.get_observation()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, required=True)
    ap.add_argument("--lamd", type=float, required=True)
    ap.add_argument("--docker_min_max", type=int, nargs=2, default=[1, 5])
    ap.add_argument("--duration", type=int, default=100, required=True)
    ap.add_argument("--output_file", type=str, required=False, default="output_file")    
    ap.add_argument("--experiment_type", type=int, required=True)
    ap.add_argument("--LGOBAL_SEED", type=int, required=False)
    
    args = ap.parse_args()
    asyncio.run(
    run(n_users=args.n_users,
        lamd= args.lamd,
        docker_min_max=args.docker_min_max,
        duration=args.duration,
        output_file=args.output_file,
        experiment_type=args.experiment_type,
        LGOBAL_SEED=args.LGOBAL_SEED
        ))
    df = pd.DataFrame(df)
    df.to_csv(f"{args.output_file}/{experiment_types[int(args.experiment_type)]}_n_user_{args.n_users}_lamda_{args.lamd}_cost.csv", index=None)

