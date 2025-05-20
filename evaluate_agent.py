import pickle
import traci
from utils import check_sumo_home
from qlearning_agent import QLearningAgent

check_sumo_home()

# Učitaj Q-tablicu
with open('qtable_ep99.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table

traci.start(["sumo-gui", "-c", "simulation/config.sumocfg"])
step = 0
TL_ID = "TL"
PHASE_DURATION = 10
last_action_time = 0

while step < 1000:
    traci.simulationStep()
    step += 1

    q_north = traci.lane.getLastStepVehicleNumber("N2TL")
    q_south = traci.lane.getLastStepVehicleNumber("S2TL")
    q_east  = traci.lane.getLastStepVehicleNumber("E2TL")
    q_west  = traci.lane.getLastStepVehicleNumber("W2TL")

    state = (q_north, q_south, q_east, q_west)

    if step - last_action_time >= PHASE_DURATION:
        action = agent.choose_action(state)
        if action == 1:
            phase = traci.trafficlight.getPhase(TL_ID)
            traci.trafficlight.setPhase(TL_ID, 1 - phase)
        last_action_time = step

traci.close()
