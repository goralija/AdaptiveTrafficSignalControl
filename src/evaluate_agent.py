# evaluate_agent.py
import pickle
import traci
from qlearning_agent import QLearningAgent
from utils import check_sumo_home, get_state
from config import TL_ID, PHASE_DURATION, CONFIG_FILE, SUMO_BINARY, MAX_STEPS

check_sumo_home()

# Load learned Q-table
with open('qtable_ep99.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table

traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
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
