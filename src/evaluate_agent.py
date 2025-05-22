# evaluate_agent.py
import pickle
import traci
from qlearning_agent import QLearningAgent
from utils import check_sumo_home, get_state
from config import TL_ID, PHASE_DURATION, CONFIG_FILE, SUMO_BINARY_EVAL, MAX_STEPS

check_sumo_home()

# Load learned Q-table
with open('q-tables-and-logs/qtable_final.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table

traci.start([SUMO_BINARY_EVAL, "-c", CONFIG_FILE])
step = 0
last_action_time = 0

while step < MAX_STEPS:
    traci.simulationStep()
    step += 1

    state = get_state()
    if step - last_action_time >= PHASE_DURATION:
        action = agent.choose_action(state)
        if action == 1:
            if action == 1:
                current_phase = traci.trafficlight.getPhase(TL_ID)
                if current_phase >= 0:
                    new_phase = (current_phase + 1) % 4  # koristi broj faza iz tvoje mreže
                    print(f"Changing phase from {current_phase} to {new_phase}")
                    traci.trafficlight.setPhase(TL_ID, new_phase)
                else:
                    print(f"Warning: Current phase is {current_phase}, skipping phase change.")
            last_action_time = step
        last_action_time = step

traci.close()
