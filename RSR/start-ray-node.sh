#!/bin/bash

if [[ "$1" == "head" ]]
then
  ray start --head --port=6379 --node-manager-port=8075 --object-manager-port=8076 --dashboard-agent-grpc-port=8077 --dashboard-agent-listen-port=8078 --metrics-export-port=8079
fi
if [[ "$1" == "worker" ]]
then
  ray start --address="$2":6379 --node-manager-port=8075 --object-manager-port=8076 --dashboard-agent-grpc-port=8077 --dashboard-agent-listen-port=8078 --metrics-export-port=8079
fi
tail -f /dev/null