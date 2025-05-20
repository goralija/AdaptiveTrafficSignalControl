import os
import sys

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        raise EnvironmentError("SUMO_HOME environment variable not set. Please add it to your shell config.")
