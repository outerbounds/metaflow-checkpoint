from collections import defaultdict
import subprocess
import json
import time
from metaflow._vendor import click
import signal
import sys
from datetime import datetime

FILE_PATH = __file__


def signal_handler(sig, frame):
    sys.exit(0)


def parse_size(size_str):
    """
    Convert human-readable size (e.g., '1K', '1M', '1G', '1T') to bytes.
    """
    size_str = size_str.strip()
    if size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1024**2)
    elif size_str.endswith("G"):
        return int(float(size_str[:-1]) * 1024**3)
    elif size_str.endswith("T"):
        return int(float(size_str[:-1]) * 1024**4)
    elif size_str.endswith("P"):
        return int(float(size_str[:-1]) * 1024**5)
    else:
        return int(size_str)  # Assume it's already in bytes if no suffix


def disk_usage_to_json():
    try:
        # Run the df command
        df_output = subprocess.check_output(["df", "-h"], text=True)

        # Split output into lines
        lines = df_output.strip().split("\n")

        # Extract headers
        headers = lines[0].split()

        # Initialize list to hold disk usage information
        disk_usage_info = []

        # Parse each line of df output after the header
        for line in lines[1:]:
            columns = line.split()
            # Map each column to its header
            usage_dict = dict(zip(headers, columns))

            # Convert the 'Size' column to bytes and add it as a new key
            if "Size" in usage_dict:
                usage_dict["SizeBytes"] = parse_size(usage_dict["Size"])

            if "Avail" in usage_dict:
                usage_dict["AvailBytes"] = parse_size(usage_dict["Avail"])

            disk_usage_info.append(usage_dict)

        # Convert the list to JSON format
        return disk_usage_info

    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


class UsageCounter:
    def __init__(self):
        self._fs_usage = defaultdict(list)

    def update(self, usage_dict):
        """
        `usage_dict` looks like. This thing will update the usage counter for each filesystem.
            {
                "Filesystem": "/dev/root",
                "Size": "582G",
                "Used": "389G",
                "Avail": "194G",
                "Use%": "67%",
                "Mounted": "/",
                "SizeBytes": 624917741568,
                "AvailBytes": 20803747840,
            },

        """
        self._fs_usage[
            usage_dict["Filesystem"] + "(Mounted: %s)" % usage_dict["Mounted"]
        ].append({**usage_dict, "Timestamp": datetime.now().isoformat()})

    def dump(self):
        return json.dumps(self._fs_usage, indent=4)


def usage_collectior(file, polling_interval):
    usage_counter = UsageCounter()
    try:
        while True:
            disk_usage_dict = disk_usage_to_json()
            if "error" in disk_usage_dict:
                print("Error getting disk usage information", file=sys.stderr)
                print(disk_usage_dict["error"], file=sys.stderr)
            else:
                for usage_dict in disk_usage_dict:
                    usage_counter.update(usage_dict)
                if file:
                    with open(file, "w") as f:
                        f.write(usage_counter.dump())
                else:
                    print(usage_counter.dump())
                time.sleep(polling_interval)
    except KeyboardInterrupt:
        pass


@click.command()
@click.option("--file", "-f", type=click.Path(), help="Output file to write JSON to")
@click.option(
    "--polling_interval",
    "--interval",
    "-i",
    type=int,
    default=5,
    help="Polling interval in seconds",
)
def run(file, polling_interval):
    signal.signal(signal.SIGINT, signal_handler)
    usage_collectior(file, polling_interval)


# Example usage
if __name__ == "__main__":
    run()
