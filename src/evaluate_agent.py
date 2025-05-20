# evaluate_agent.py
import pickle
import traci
from qlearning_agent import QLearningAgent
from utils import check_sumo_home
from simulation_utils import get_state

check_sumo_home()

TL_ID = "n0"
PHASE_DURATION = 10
MAX_STEPS = 1000
CONFIG_FILE = "simulation/config.sumocfg"

# Load learned Q-table
with open('qtable_ep99.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table

traci.start(["sumo-gui", "-c", CONFIG_FILE])
step = 0
last_action_time = 0

while step < MAX_STEPS:
    traci.simulationStep()
    step += 1

    state = get_state()
    if step - last_action_time >= PHASE_DURATION:
        action = agent.choose_action(state)
        if action == 1:
            phase = traci.trafficlight.getPhase(TL_ID)
            traci.trafficlight.setPhase(TL_ID, 1 - phase)
        last_action_time = step

traci.close()
