#!/bin/bash
#
# start_everything.sh
# Prompts, then launches health monitor, watchdog, miners with resume logic,
# and a status loop—all in one terminal.

set -e

### 0) Cleanup old processes and files
echo "🚀 Cleaning up old processes and files…"
pkill -f "seeds.*--workers" || true
pkill -f "gpu_health_monitor.sh" || true
pkill -f "launch_watchdog.sh" || true
rm -f miner*.log restart.log gpu_health.log worker*.offset found_seeds.txt overheated.flag nohup.out
sleep 1

### 1) Prompt the user
# Prompt for seed words
read -r -p "Enter known seed words (space-separated, or leave blank if none): " KNOWN_SEED_WORDS

# Validate seed words
if [[ -n "$KNOWN_SEED_WORDS" ]]; then
  KNOWN_COUNT=$(echo "$KNOWN_SEED_WORDS" | wc -w)
  if [[ "$KNOWN_COUNT" -gt 12 ]]; then
    echo "❌ Too many seed words: $KNOWN_COUNT (max 12)"
    exit 1
  fi
  while read -r word; do
    if ! grep -Fx "$word" words.txt > /dev/null; then
      echo "❌ Invalid seed word: '$word' not in words.txt"
      exit 1
    fi
  done <<< "$(echo "$KNOWN_SEED_WORDS" | tr ' ' '\n')"
fi

# Prompt for target address
read -r -p "Enter target Ethereum address (hex, no 0x; leave blank for zero mode): " TARGET_ADDRESS

# Validate address
if [[ -n "$TARGET_ADDRESS" ]]; then
  if [[ ! "$TARGET_ADDRESS" =~ ^[0-9a-fA-F]{0,40}$ ]]; then
    echo "❌ Invalid address: must be 0–40 hex chars (no 0x)"
    exit 1
  fi
fi

# Try to detect number of GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo 0)
if [[ "$GPU_COUNT" -gt 0 ]]; then
  echo "🎯 Detected $GPU_COUNT GPU(s)."
else
  echo "⚠️ No GPUs detected via nvidia-smi. Defaulting to 1 worker."
  GPU_COUNT=1
fi

# Ask user for number of workers, suggest GPU count
read -r -p "How many workers to run? [default: $GPU_COUNT] " USER_WORKERS
TOTAL_WORKERS=${USER_WORKERS:-$GPU_COUNT}

# Validate number of workers
if [[ ! "$TOTAL_WORKERS" =~ ^[0-9]+$ ]] || [[ "$TOTAL_WORKERS" -lt 1 ]]; then
  echo "❌ Invalid number of workers: $TOTAL_WORKERS"
  exit 1
fi

### 2) Generate helper scripts

# GPU health monitor (exits on found or overheat)
cat << 'EOF' > gpu_health_monitor.sh
#!/bin/bash
DANGER_TEMP=85
CHECK_INTERVAL=30
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  for t in $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null); do
    (( t >= DANGER_TEMP )) && {
      echo "🔥 GPU overheat (${t}C)! Shutting down miners."
      pkill -f "seeds.*--workers" || true
      touch overheated.flag
      exit 0
    }
  done
  sleep $CHECK_INTERVAL
done
EOF
chmod +x gpu_health_monitor.sh

# Watchdog (crash & slow detection, with resume)
cat << 'EOF' > launch_watchdog.sh
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
EOF
chmod +x launch_watchdog.sh

### 3) Launch helper scripts
echo "🛡️ Starting GPU Health Monitor..."
nohup ./gpu_health_monitor.sh > gpu_health.log 2>&1 &
echo "🛡️ Starting Watchdog..."
nohup ./launch_watchdog.sh > watchdog.log 2>&1 &

### 4) Spawn miners with resume offsets
echo "🔥 Spawning $TOTAL_WORKERS miners…"
for ((i=0; i<TOTAL_WORKERS; i++)); do
  OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
  echo "Starting worker $i from offset $OFFSET..."
  CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
    --workers $TOTAL_WORKERS \
    --known "$KNOWN_SEED_WORDS" \
    --address "$TARGET_ADDRESS" \
    --wordlist words.txt \
    > miner$i.log 2>&1 &
done

### 5) Status loop (foreground)
UNKNOWN=$((12 - $(echo "$KNOWN_SEED_WORDS" | wc -w)))
TOTAL_SPACE=$(echo "2048^$UNKNOWN" | bc)
BATCH=$(echo "$TOTAL_SPACE / $TOTAL_WORKERS" | bc)
echo "Total search space: $TOTAL_SPACE, per worker: $BATCH"

echo
echo "✅ All miners running. Showing status every 30 s (Ctrl-C to exit):"
echo

while true; do
  [[ -f found_seeds.txt ]] && {
    echo "✅ Seed found: $(cat found_seeds.txt)"
    exit 0
  }
  [[ -f overheated.flag ]] && {
    echo "❌ GPU overheated. Check gpu_health.log."
    exit 1
  }
  echo "----- Status @ $(date) -----"
  ALL_DONE=1
  for ((i=0; i<TOTAL_WORKERS; i++)); do
    tested=$(grep -oP 'tested \K[0-9]+/[0-9]+' miner$i.log | tail -1 | cut -d'/' -f1 || echo 0)
    speed=$(grep -oP 'seeds/sec.*\K[0-9]+(\.[0-9]+)?' miner$i.log | tail -1 || echo "N/A")
    if [[ ! "$tested" =~ ^[0-9]+$ ]]; then
      tested=0
    fi
    rem=$(echo "$BATCH - $tested" | bc 2>/dev/null || echo "$BATCH")
    if [[ "$rem" -lt 0 ]]; then
      rem=0
    fi
    if [[ "$tested" -lt "$BATCH" ]]; then
      ALL_DONE=0
    fi
    # Check if worker is running
    if ! pgrep -f "seeds.*--workers $TOTAL_WORKERS.*$i.log" > /dev/null; then
      echo "Worker $i: crashed or not started, check miner$i.log"
    else
      printf "Worker %d: tested %'d/%'d (remaining %'d), %s seeds/sec\n" \
        "$i" "$tested" "$BATCH" "$rem" "$speed"
    fi
  done
  echo
  [[ "$ALL_DONE" -eq 1 ]] && {
    echo "❌ Search completed; no matching mnemonic found."
    exit 1
  }
  sleep 30
done