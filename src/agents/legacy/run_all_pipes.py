import subprocess
import os
from datetime import datetime
import sys

# List of Python scripts to run
scripts = [
    "batch",
    "context_analysis",
    "neighbors",
    "no_context"
]

# Ensure log directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate timestamp for logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sys.stdout.reconfigure(encoding='utf-8') 
for script in scripts:
    log_file = os.path.join(log_dir, f"log_{script}_{timestamp}.txt")
    with open(log_file, "w") as f:
        print(f"Running {script}...")
        process = subprocess.run(["python","-m", "src."+script], stdout=f, stderr=f, encoding="utf-8")
        print(f"Finished {script} with return code {process.returncode}")