#!/bin/bash
N=4
OS=$(uname)

echo "Killing processes for $N terminal containers ..."
for i in $(seq 1 $N); do
    if [[ "$OS" == "Darwin" ]]; then
        pids=$(pgrep -f "mec_simulation_$i")
        if [ -n "$pids" ]; then
            kill -9 $pids
        fi
osascript <<EOF
tell application "Terminal"
    repeat with i from (count windows) to 1 by -1
        set w to window i
        if custom title of w contains "mec_simulation_$i" then
            close w saving no
        end if
    end repeat
end tell
EOF
    else
        pids=$(ps -ef | grep "mec_simulation_$i" | grep -v grep | awk '{print $2}')
        if [ -n "$pids" ]; then
            echo "Killing PID(s): $pids"
            kill -9 $pids
        fi
    fi
done

echo "closing docker containers ..."
if [ "$(docker ps -q)" ]; then
    docker stop $(docker ps -q)
fi

if [ "$(docker ps -a -q)" ]; then
    docker rm $(docker ps -a -q)
fi
echo "All Docker containers stopped and removed."
