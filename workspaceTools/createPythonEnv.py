import subprocess
import sys
import os

# get the base of the directory
try:
    cmd = "git rev-parse --show-toplevel"
    base_dir = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, shell=True
    ).strip().decode('ascii')

except subprocess.CalledProcessError as e:
    print("Subprocess failed with return code: ", e.returncode, " and output: ", e.output)
    exit(1)

# make an venv file if not exists
if not os.path.exists(os.path.join(base_dir + '/venv/')) :
    print("Creating python env")
    cmd = "python -m venv venv"
    subprocess.check_call(
        cmd,
        cwd=base_dir,
        shell=True
    )
