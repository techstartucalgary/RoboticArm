import subprocess
import os

if not os.path.isfile("requirements.txt"):
    print("requirements.txt not found")
    exit(1)

cmd = "pip install -r requirements.txt"
try:
    subprocess.check_call(
        cmd, 
        shell=True
    )

except subprocess.CalledProcessError as e:
    print(e.output)
    exit(1)