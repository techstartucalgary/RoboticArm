import subprocess
import os

if not os.path.isfile("requirements.txt"):
    print("requirements.txt not found")
    exit(1)

cmd = "pip install -r requirements.txt"
subprocess.check_call(
    cmd, 
)