#!/bin/bash
DANGER_TEMP=85
CHECK_INTERVAL=30
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  for t in $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); do
    (( t >= DANGER_TEMP )) && {
      echo "🔥 GPU overheat (${t}C)! Shutting down miners."
      killall seeds || true
      touch overheated.flag
      exit 0
    }
  done
  sleep $CHECK_INTERVAL
done
