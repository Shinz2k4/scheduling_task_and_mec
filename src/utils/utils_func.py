import subprocess
import re
import os
import json

N_SERVER = 4


def parse_size(value_str):
    """
     GiB
     B, KiB, MiB, GiB, TiB, kB, MB, GB, TB ---> GiB
    """
    value_str = value_str.strip()
    try:
        # Binary units
        if value_str.endswith("KiB"):
            num = float(value_str.replace("KiB", ""))
            return num / (1024**2)  # GiB
        elif value_str.endswith("MiB"):
            num = float(value_str.replace("MiB", ""))
            return num / 1024
        elif value_str.endswith("GiB"):
            num = float(value_str.replace("GiB", ""))
            return num
        elif value_str.endswith("TiB"):
            num = float(value_str.replace("TiB", ""))
            return num * 1024

        # Decimal units
        elif value_str.endswith("kB"):
            num = float(value_str.replace("kB", ""))
            return num / (1000**3)
        elif value_str.endswith("MB"):
            num = float(value_str.replace("MB", ""))
            return num / 1000
        elif value_str.endswith("GB"):
            num = float(value_str.replace("GB", ""))
            return num
        elif value_str.endswith("TB"):
            num = float(value_str.replace("TB", ""))
            return num * 1000

        # Bytes
        elif value_str.endswith("B"):
            num = float(value_str.replace("B", ""))
            return num / (1024**3)

        # Không có đơn vị, giả sử là GiB
        else:
            return float(value_str)

    except Exception as e:
        print(f"Warning: cannot parse '{value_str}': {e}")
        return 0.0

def get_docker_metrics_by_name():
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        return []
    container_names = [name for name in result.stdout.splitlines() if name]
    container_names.sort(key=lambda name: int(re.search(r'(\d+)$', name).group(1) if re.search(r'(\d+)$', name) else 0))
    try:
        stats = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
            "{{.Name}} {{.CPUPerc}} {{.MemPerc}} {{.MemUsage}} {{.NetIO}}"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except Exception as e:
        print(f"Error running docker stats: {e}")
    except:
        print("Error running docker stats")

    if stats.stdout == "":
        return []

    values = []
    for container_name in container_names:
        for line in stats.stdout.splitlines():
            if not line.startswith("mec"):
                continue
            parts = line.split()
            if parts[0] != container_name:
                continue
            # print(parts)
            try:
                # parts = [Name, CPU%, MEM%, MemUsage, ..., NetIO1, ..., NetIO2, ...]
                values.append(float(parts[1].strip('%')))
                values.append(float(parts[2].strip('%')))
                values.append(parse_size(parts[3]))  # MemUsage
                values.append(parse_size(parts[5]))  # NetIO1
                values.append(parse_size(parts[6]))  # NetIO2
                values.append(parse_size(parts[8]))  # NetIO2
            except Exception as e:
                print(f"Error parsing line {line}: {e}")
    # print("Docker metrics values:", values )
    return values

def thread_func(container):
    container["metrics"] = get_docker_metrics_by_name()

def get_active_service():
    service_dir = "./../service"
    list_service_in_docker = {}
    for filename in os.listdir(service_dir):
        match = re.match(r"active_services_in_docker_(\d+)-th\.json", filename)
        if match:
            docker_id = int(match.group(1))
            filepath = os.path.join(service_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                list_service_in_docker[docker_id] = list(data.keys())
    list_docker = []
    if len(list_service_in_docker) >0:
        list_docker = list(list_service_in_docker.keys())
    
    return list_docker, list_service_in_docker


def get_feature(obs, id_picture, model, docker):
    #model index not title
    chunk_size = len(obs) // N_SERVER
    start = (docker-1) * chunk_size
    end = start + chunk_size
    feature = [id_picture, model, docker+1]
    feature += obs[start:end]
    return feature

def get_feature_delay(obs, id_picture, model_id, slected_docker):
    chunk_size = len(obs) // N_SERVER
    start = (slected_docker - 1) * chunk_size
    end = slected_docker * chunk_size
    feature = [id_picture, model_id, slected_docker]
    feature += obs[start:end-1]
    return feature
