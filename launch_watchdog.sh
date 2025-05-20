#!/bin/bash
THRESHOLD=300000  # 300K seeds/sec, realistic for Quadro K2200
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  sleep 30
  for ((i=0; i<TOTAL_WORKERS; i++)); do
    [[ -f found_seeds.txt ]] && exit 0

    # Crash detection:
    if ! pgrep -f "seeds.*--workers $TOTAL_WORKERS.*--known.*$i.log" > /dev/null; then
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
      | grep -oP 'seeds/sec.*\K[0-9]+(\.[0-9]+)?' | tail -1)
    if [[ -n "$speed" && $(echo "$speed < $THRESHOLD" | bc -l) -eq 1 ]]; then
      echo "⚠️ Worker $i slow ($speed seeds/sec); restarting from offset…" | tee -a restart.log
      pkill -f "seeds.*--workers $TOTAL_WORKERS.*$i.log" || true
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
