"""
Robust Runner for Relabeling Script

Features:
- Checks Ollama is running before starting
- Auto-restarts if script crashes
- Hourly progress reports
- Restarts Ollama if unresponsive
- Logs everything to file

Usage:
    python utils/run_relabeling.py

    # With custom settings
    NUM_WORKERS=6 python utils/run_relabeling.py
"""

import subprocess
import time
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIG
# ============================================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"

OUTPUT_DIR = DATA_DIR / "LLM-relabeled-data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_FILE = OUTPUT_DIR / "runner.log"

# How often to check progress (seconds)
PROGRESS_CHECK_INTERVAL = 3600  # 1 hour

# Max consecutive failures before giving up
MAX_FAILURES = 5

# Wait time between restarts (seconds)
RESTART_WAIT = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(message):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    # Append to log file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_ollama():
    """Check if Ollama is running and responsive"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"Ollama check failed: {e}")
        return False


def wait_for_ollama(max_wait=300):
    """Wait for Ollama to become available"""
    log("Waiting for Ollama...")
    start = time.time()

    while time.time() - start < max_wait:
        if check_ollama():
            log("Ollama is ready")
            return True
        time.sleep(10)

    log(f"Ollama not available after {max_wait}s")
    return False


def restart_ollama():
    """Try to restart Ollama service"""
    log("Attempting to restart Ollama...")
    try:
        # Try to kill existing ollama
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        time.sleep(5)

        # Start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(10)

        # Wait for it to be ready
        return wait_for_ollama()
    except Exception as e:
        log(f"Failed to restart Ollama: {e}")
        return False


def get_progress():
    """Get current progress from checkpoints"""
    if not CHECKPOINT_DIR.exists():
        return {"total_processed": 0, "valid": 0, "rejected": 0, "files_done": 0}

    total_processed = 0
    total_valid = 0
    total_rejected = 0
    files_done = 0

    for checkpoint_file in CHECKPOINT_DIR.glob("*_checkpoint.json"):
        try:
            with open(checkpoint_file) as f:
                data = json.load(f)

            processed = data.get("processed", 0)
            valid = len(data.get("valid", [])) + len(data.get("fixed", []))
            rejected = len(data.get("rejected", [])) + len(data.get("errors", []))

            total_processed += processed
            total_valid += valid
            total_rejected += rejected

            # Check if file is complete (rough heuristic)
            if processed > 0 and valid + rejected >= processed:
                files_done += 1

        except Exception:
            pass

    return {
        "total_processed": total_processed,
        "valid": total_valid,
        "rejected": total_rejected,
        "files_done": files_done
    }


def print_progress():
    """Print current progress summary"""
    progress = get_progress()

    log("=" * 50)
    log("HOURLY PROGRESS REPORT")
    log("=" * 50)
    log(f"  Samples processed: {progress['total_processed']:,}")
    log(f"  Valid samples:     {progress['valid']:,}")
    log(f"  Rejected samples:  {progress['rejected']:,}")
    log(f"  Files completed:   {progress['files_done']}")

    # Estimate remaining time
    if progress['total_processed'] > 0:
        # Check how long we've been running from log file
        try:
            with open(LOG_FILE) as f:
                first_line = f.readline()
            start_time = datetime.strptime(first_line[1:20], "%Y-%m-%d %H:%M:%S")
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            rate = progress['total_processed'] / elapsed if elapsed > 0 else 0
            remaining = (125591 - progress['total_processed']) / rate if rate > 0 else 0
            log(f"  Processing rate:   {rate:.0f} samples/hour")
            log(f"  Estimated remaining: {remaining:.1f} hours")
        except Exception:
            pass

    log("=" * 50)


def run_relabeling_script():
    """Run the main relabeling script"""
    script_path = SCRIPT_DIR / "relabel_training_data.py"

    env = os.environ.copy()
    env["OLLAMA_URL"] = OLLAMA_URL
    env["OLLAMA_MODEL"] = MODEL

    log(f"Starting relabeling script...")
    log(f"  Model: {MODEL}")
    log(f"  Workers: {env.get('NUM_WORKERS', '4')}")

    process = subprocess.Popen(
        ["python", str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return process


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    log("=" * 60)
    log("ROBUST RELABELING RUNNER")
    log("=" * 60)

    # Check initial state
    initial_progress = get_progress()
    if initial_progress['total_processed'] > 0:
        log(f"Resuming from checkpoint: {initial_progress['total_processed']:,} samples already processed")

    # Check Ollama
    if not check_ollama():
        log("Ollama not running, attempting to start...")
        if not restart_ollama():
            log("ERROR: Could not start Ollama. Please run 'ollama serve' manually.")
            return

    log(f"Ollama ready at {OLLAMA_URL}")

    # Main loop
    consecutive_failures = 0
    last_progress_check = time.time()

    while consecutive_failures < MAX_FAILURES:
        # Start the script
        process = run_relabeling_script()

        # Monitor the process
        while True:
            # Check if process is still running
            return_code = process.poll()

            if return_code is not None:
                # Process ended
                if return_code == 0:
                    log("Relabeling script completed successfully!")
                    print_progress()
                    return
                else:
                    log(f"Script exited with code {return_code}")
                    consecutive_failures += 1
                    break

            # Read output (non-blocking)
            try:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
            except Exception:
                pass

            # Hourly progress check
            if time.time() - last_progress_check > PROGRESS_CHECK_INTERVAL:
                print_progress()
                last_progress_check = time.time()

                # Also check Ollama health
                if not check_ollama():
                    log("Ollama became unresponsive!")
                    process.terminate()
                    time.sleep(5)

                    if restart_ollama():
                        log("Ollama restarted, resuming...")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                    break

            time.sleep(1)

        # Wait before restart
        if consecutive_failures < MAX_FAILURES:
            log(f"Waiting {RESTART_WAIT}s before restart... (failure {consecutive_failures}/{MAX_FAILURES})")
            time.sleep(RESTART_WAIT)

            # Check Ollama before restart
            if not check_ollama():
                if not restart_ollama():
                    consecutive_failures += 1
                    continue

    log(f"ERROR: Too many failures ({MAX_FAILURES}). Giving up.")
    log("Check the logs and checkpoints, then restart manually.")
    print_progress()


if __name__ == "__main__":
    main()
