import subprocess
import sys
from datetime import datetime


def print_colored(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def run_script(script_path, params):
    run_name, learning_rate, n_estimators = params

    print_colored(f"\n{'='*60}", "blue")
    print_colored(f"Starting run: {run_name}", "blue")
    print_colored(
        f"Parameters: learning_rate={learning_rate}, n_estimators={n_estimators}",
        "blue",
    )
    print_colored(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "blue")
    print_colored(f"{'='*60}\n", "blue")

    try:
        subprocess.run([sys.executable, script_path] + params, check=True)
        print_colored(f"\n{'='*60}", "green")
        print_colored(f"Run completed successfully: {run_name}", "green")
        print_colored(
            f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "green"
        )
        print_colored(f"{'='*60}\n", "green")
    except subprocess.CalledProcessError as e:
        print_colored(f"\n{'='*60}", "red")
        print_colored(f"An error occurred during run: {run_name}", "red")
        print_colored(f"Error details: {e}", "red")
        print_colored(f"{'='*60}\n", "red")


# List of parameters
params = [
    ["BIN-125", "0.1", "1500"],
    ["BIN-124", "0.1", "1500"],
    ["BIN-123", "0.1", "1500"],
    ["BIN-106", "0.05", "3000"],
    ["BIN-103", "0.05", "3000"],
    ["BIN-102", "0.05", "1500"],
    ["BIN-101", "0.1", "1500"],
    ["BIN-100", "0.1", "600"],
    ["BIN-99", "0.1", "300"],
    ["BIN-98", "0.1", "10"],
    ["BIN-97", "0.1", "100"],
]

# Path to your main script
script_path = "script.py"

# Run the script for each set of parameters
for param in params:
    run_script(script_path, param)
