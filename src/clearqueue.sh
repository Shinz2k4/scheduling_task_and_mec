#!/bin/bash

# Số lượng docker
NUM_DOCKERS=4
PORT_BASE=10000  

for i in $(seq 1 $NUM_DOCKERS); do
    PORT=$((PORT_BASE + i))
    echo "Clearing queue for docker $i at port $PORT"
    ../.venv/bin/python3 host_send_request.py --request clearq --num 1 --port $PORT --docker $i
done
