# simulation_utils.py
import traci

def get_state():
    q_north = traci.lane.getLastStepVehicleNumber("e2_0")
    q_south = traci.lane.getLastStepVehicleNumber("e4_0")
    q_east  = traci.lane.getLastStepVehicleNumber("e6_0")
    q_west  = traci.lane.getLastStepVehicleNumber("e0_0")
    return (q_north, q_south, q_east, q_west)
