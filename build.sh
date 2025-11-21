#!/bin/bash

docker build -f docker/Dockerfile -t mec_simulation:latest .

N_SERVERS=4
BASE_PORT=10000
OS_TYPE=$(uname)

mkdir -p ./tmp/docker_sockets

if [[ "$OS_TYPE" == "Linux" ]]; then
    TOTAL_CORES=$(nproc)
    MAX_CORES_PER_DOCKER=2
    cores=($(seq 0 $((TOTAL_CORES-1))))

    cores=($(seq 0 $((TOTAL_CORES-1))))

    docker_cores_list=("dumy")
    start_index=0
    for ((i=1; i<=N_SERVERS; i++)); do
        selected_cores=("${cores[@]:start_index:MAX_CORES_PER_DOCKER}")
        docker_cores_list+=("$(IFS=,; echo "${selected_cores[*]}")")
        start_index=$((start_index + MAX_CORES_PER_DOCKER))
    done
fi
for i in $(seq 1 $N_SERVERS); do
    container_name="mec_simulation_$i"
    port=$((BASE_PORT + i))  

    echo "Starting container $container_name on port $port ..."
    if [[ "$OS_TYPE" == "Linux" ]]; then
        echo "Detected Linux "
        echo "Launching $container_name on cores ${docker_cores_list[$i]}"
        docker run -dit \
            --network=host \
            --cpuset-cpus="${docker_cores_list[$i]}" \
            --cpus="$(echo ${docker_cores_list[$i]} | awk -F',' '{print NF}')" \
            --memory=5g \
            --memory-swap=5g \
            -v ./val2017:/shared \
            -v ./service:/service \
            -v ./tmp/docker_sockets:/tmp/docker_sockets\
            --name "$container_name" \
            mec_simulation:latest

    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        docker run -dit \
            -p $port:$port \
            -v ./val2017:/shared \
            -v ./service:/service \
            -v ./tmp/docker_sockets:/./tmp/docker_sockets \
            --name "$container_name" \
            mec_simulation:latest
    else
        echo "Unsupported OS: $OS_TYPE."
    fi
            

    echo "Sending install request to container $i ..."
done
wait
for i in $(seq 1 $N_SERVERS); do
    container_name="mec_simulation_$i"
    port=$((BASE_PORT + i))
    echo "Starting uvicorn inside container $container_name on port $port ..."
    if [[ "$OS_TYPE" == "Darwin" ]]; then
osascript <<EOF
    tell application "Terminal"
        do script "docker exec -it  $container_name bash -c 'export CONTAINER_PORT=$port && cd src && uvicorn servers.handle_host_request:app --host 0.0.0.0 --port $port'"
        set custom title of front window to "mec_simulation_$i"
    end tell
EOF
    elif  [[ "$OS_TYPE" == "Linux" ]]; then
        gnome-terminal --title="mec_simulation_$i" -- bash -c "docker exec -it $container_name bash -c 'export CONTAINER_PORT=$port && cd src && uvicorn servers.handle_host_request:app --host 0.0.0.0 --port $port'; exec bash"
    else
        echo "Unsupported OS: $OS_TYPE "
    fi
done
wait
sleep 3
for i in $(seq 1 $N_SERVERS); do
    port=$((BASE_PORT + i))
    echo "Sending install request to docker $i on port $port ..."
    cd src
    ../.venv/bin/python3 -m host_send_request \
        --request install \
        --num 1 \
        --docker $i \
        --port $port
    cd ..
done
wait
echo "All install requests sent."