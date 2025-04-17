#!/bin/bash
while true; do
  current_time=$(date "+%H:%M:%S")
  echo "[$current_time]"
  mpstat | awk '$12 ~ /[0-9.]+/ { print "CPU Usage: " 100 - $12"%" }'
  free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
  echo "----------------------------"
  sleep 1
done
