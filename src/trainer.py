import asyncio, argparse
import random
import httpx
import numpy as np
import re, sys, subprocess
from env import OffloadingEnv
from ddqn_agent import DDQNAgent
import matplotlib.pyplot as plt
import matplotlib
from predict_required_cpu import CPUPredictor
matplotlib.use("Agg") 
import time
from utils.utils_func import get_active_service, get_feature, get_feature_delay
import os
import torch
from trainer_processing_time import DelayPredictor
import joblib
from fdo import fast_detect_outage
import pandas as pd
N_SERVER = 4

df = {"id": [], "id_picture":[], "predict_cost": [] }
experiment_types = {
    0: 'random',
    1: 'drl_train',
    2: 'drl_prediction',
    3: 'esimated_processing_time',
    4: 'drl_train_with_history_task_observation',
    5: 'drl_prediction_with_history_task_observation',
    6: 'fast_detect_outage'
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
    print('state init')
    obs = await env.get_observation()
    queue = {}
    rewards = []
    all_reward = {}
    queues = [[] for _ in range(N_SERVER)]
    agent = DDQNAgent(len(obs)+1, N_SERVER, batch_size = 512)
    if experiment_types[experiment_type].find('drl_prediction')!=-1:
        print("loading model for drl prediction")
        agent.load(drl_checkpoint)
    done = False
    save_dir = "train_result"
    fearture_vecs = get_feature(obs, id_picture=0, model=0, docker=1) #just for get size
    if experiment_types[experiment_type] == 'esimated_processing_time':
        print("loading model to estimate processing time")
        load_model_estimate_processing_time = DelayPredictor(input_dim=len(fearture_vecs))
        load_model_estimate_processing_time.load_model(f"{save_dir}/pretrained_processing_estimation.pth")
    check_done  = 0
    if experiment_types[experiment_type] == 'esimated_processing_time':
        print("loading scaler for esimated_processing_time")
        scaler = joblib.load(f"{save_dir}/scaler.pkl")
    
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
        feature_vector_delay = np.array(get_feature_delay(obs, id_picture, model_id, 1)) # so 1 o day la dai dien cho slected docker vi 4 docker nhu 1 ( cung cau hinh ) 
        feature_vector_delay = feature_vector_delay.reshape(1, -1)

        required_cpu = 0
       
        task_deadline = round(random.uniform(1, 2), 2)

        if experiment_types[experiment_type] == 'random':
            print("running random experiment")
            slected_docker = rng.integers(docker_min_max[0], docker_min_max[1])
            predict_cost = time.perf_counter() - start_time
        elif experiment_types[experiment_type].find('drl')!=-1:
            print("running drl experiment")
            obs_pre = obs.copy()
            obs_pre.insert(0, model_id)
            slected_docker = agent.act(obs_pre) + 1
            predict_cost = time.perf_counter() - start_time
        elif experiment_types[experiment_type] == 'esimated_processing_time':
            print("running esimated_processing_time experiment")
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
        elif experiment_types[experiment_type] == 'fast_detect_outage':
            print("running fast_detect_outage experiment")
            Predictor_cpu_required = CPUPredictor(feature_vector_delay)
            required_cpu = Predictor_cpu_required.predict_required_cpu()
            cur_queues = await env.get_queues()
            delay_min = 999999999999999999
            flag_min = 999999999999999999
            flag_success = False
            for i in range(N_SERVER):
                flag, delay = fast_detect_outage(cur_queues[i], cnt, required_cpu, task_deadline)
                if flag <= flag_min: 
                    if delay < delay_min :
                        delay_min = delay
                        flag_min = i
                        slected_docker = i + 1
                        flag_success = True
            
            predict_cost = time.perf_counter() - start_time  
            if not flag_success:
                
                raise ValueError("Task không thể lập lịch được theo FDO")      
        df["id"].append(cnt)
        df["id_picture"].append(id_picture)
        df['predict_cost'].append(predict_cost)
        slected_port = port_base + slected_docker
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
            "--required_cpu", str(required_cpu),
            "--deadline", str(task_deadline)
        ]
        
        
        t1= time.perf_counter()
        env.update_historical_tasks(slected_docker - 1, model_id)
        subprocess.Popen(cmd)
        next_state = await env.get_observation()
        next_state_ = next_state.copy()
        next_state_.insert(0, model_id)
        obs.insert(0, model_id)
        queue[cnt] = [obs, slected_docker - 1, next_state_]
        obs = next_state
        reward = await env.get_reward()
        all_reward.update(reward)
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
                    if re_val != "None" and re_val is not None:
                        train_data = queue[int(taskid)]
                        if type(train_data) == str:
                            continue
                        train_data.append(-float(re_val))
                        if re_val > 50 or (cnt > 0 and cnt%200 == 0 and experiment_types[experiment_type] == 'drl_train'):
                            done = True
                            agent.remember(train_data[0], train_data[1], train_data[2], train_data[3], True)
                        else:
                            agent.remember(train_data[0], train_data[1], train_data[2], train_data[3], False)
                        rewards.append(-float(re_val))
                        del queue[int(taskid)]
                        del_r_key.append(taskid)  
                        print(f"---------> {re_val}")
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
            agent.save(drl_checkpoint)

            payload = {"name": f"{output_file}/results_n_server_{N_SERVER}_n_user_{n_users}_{experiment_types[experiment_type]}_seed_{LGOBAL_SEED}.csv"}

            url = "http://127.0.0.1:15000/restart_saver_no_reset_df"
            try:
                r = httpx.post(url, json=payload, timeout=30)
                print(r.status_code, r.text)
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
                task_deadline = round(random.uniform(1, 2), 2)
                Predictor_cpu_required = CPUPredictor(feature_vector_delay)
                required_cpu = Predictor_cpu_required.predict_required_cpu()
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
                    "--required_cpu", str(required_cpu),
                    "--deadline", str(task_deadline)
                ]
                queue[cnt] = "not record"
                env.update_historical_tasks(slected_docker - 1, model_id)
                subprocess.Popen(cmd)
                cnt += 1
                print(f"Conpensate task id {cnt}"   )
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

