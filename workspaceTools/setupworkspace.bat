@echo off
for /f %%i in ('git rev-parse --show-toplevel') DO SET "WORKSPACE_BASE=%%i"

python.exe %WORKSPACE_BASE%/workspaceTools/createPythonEnv.py
call %WORKSPACE_BASE%/winVenv/Scripts/activate.bat
python.exe %WORKSPACE_BASE%/workspaceTools/installRequirements.py
