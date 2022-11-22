
WORKSPACE_BASE=`git rev-parse --show-toplevel`

python $WORKSPACE_BASE/workspaceTools/createPythonEnv.py
source $WORKSPACE_BASE/linuxVenv/bin/activate
python $WORKSPACE_BASE/workspaceTools/installRequirements.py

