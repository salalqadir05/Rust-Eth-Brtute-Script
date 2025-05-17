# Ethereum GPU Brute-Force Recovery Tool

This project uses GPU acceleration to recover an Ethereum wallet seed phrase by brute-force. It includes watchdog and GPU health monitoring scripts for automated handling and miner restarts.

---

## 🚀 Build Instructions

Build the project in **release** mode for optimal performance:

```bash
cargo build --release
```

# Remove old symlink or script (ignore errors if not present)
```rm -f start_everything```

# Create a symbolic link: start_everything → start_everything.sh
```ln -s start_everything.sh start_everything```

# Make sure the script is executable
```chmod +x start_everything.sh```

# Final Run This 
./start_everything
