#!/bin/bash
THRESHOLD=50000000  # 50M seeds/sec
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  sleep 60
  for ((i=0; i<TOTAL_WORKERS; i++)); do
    [[ -f found_seeds.txt ]] && exit 0

    # Crash detection:
    if ! pgrep -f "seeds.*--workers $TOTAL_WORKERS.* worker$i.offset" > /dev/null; then
      echo "💥 Worker $i crashed; resuming from offset…" | tee -a restart.log
      OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
      CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
        --workers $TOTAL_WORKERS \
        --known "$KNOWN_SEED_WORDS" \
        --address "$TARGET_ADDRESS" \
        --wordlist words.txt \
        > miner$i.log 2>&1 &
      continue
    fi

    # Slow miner detection:
    speed=$(tail -n20 miner$i.log 2>/dev/null \
      | grep -oP 'Speed: \K[0-9,]+' | tr -d ',')
    if [[ -n "$speed" && $speed -lt $THRESHOLD ]]; then
      echo "⚠️ Worker $i slow ($speed); restarting from offset…" | tee -a restart.log
      pkill -f "seeds.*--workers $TOTAL_WORKERS" || true
      sleep 2
      OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
      CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
        --workers $TOTAL_WORKERS \
        --known "$KNOWN_SEED_WORDS" \
        --address "$TARGET_ADDRESS" \
        --wordlist words.txt \
        > miner$i.log 2>&1 &
    fi
  done
done
