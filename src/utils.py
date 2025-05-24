import os
import sys
import traci
import subprocess
from config import TL_ID, CONFIG_FILE, SUMO_BINARY_EVAL, MAX_STEPS

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        raise EnvironmentError("SUMO_HOME environment variable not set. Please add it to your shell config.")

def get_state():
    q_north = traci.lane.getLastStepVehicleNumber("e2_0")
    q_south = traci.lane.getLastStepVehicleNumber("e4_0")
    q_east  = traci.lane.getLastStepVehicleNumber("e6_0")
    q_west  = traci.lane.getLastStepVehicleNumber("e0_0")
    return (q_north, q_south, q_east, q_west)

def generate_random_routes():
    subprocess.run([
        "python", 
        f"{os.environ['SUMO_HOME']}/tools/randomTrips.py",
        "-n", "network.net.xml",
        "-r", "routes.rou.xml",
        "-b", "0",
        "-e", "100",
        "-p", "1",
        "--validate"
    ])

def get_phase_count():
    num_phases = traci.trafficlight.getAllProgramLogics(TL_ID)[0].getPhases()
    return len(num_phases)