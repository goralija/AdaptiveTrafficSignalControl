import os
import sys
import traci

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