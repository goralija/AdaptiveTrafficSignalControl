# run_training.py
import time
import pickle
import os
import shutil
import traci
from qlearning_agent import QLearningAgent
from utils import check_sumo_home, get_state
from config import TL_ID, PHASE_DURATION, CONFIG_FILE, SUMO_BINARY, MAX_STEPS

check_sumo_home()

NUM_EPISODES = 100

agent = QLearningAgent(actions=[0, 1])  # 0: keep, 1: change

# clear the directory 'q-tabele-i-logovi' if it exists
try:
    if os.path.exists("q-tabele-i-logovi"):
        shutil.rmtree("q-tabele-i-logovi")
    os.makedirs("q-tabele-i-logovi")
except FileExistsError:
    print("Directory already exists, skipping creation.")
except FileNotFoundError:
    print("Directory not found, creating a new one.")
    os.makedirs("q-tabele-i-logovi")
except PermissionError:
    print("Permission denied, unable to create directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    

def run_episode(episode):
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    step = 0
    last_action_time = 0
    total_reward = 0

    state = get_state()

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        current_phase = traci.trafficlight.getPhase(TL_ID)
        print(f"Step {step} - Current phase: {current_phase}")

        if current_phase == -1:
            # Semafor nije spreman, ne menjaj fazu i samo nastavi
            continue

        if step - last_action_time >= PHASE_DURATION:
            action = agent.choose_action(state)
            if action == 1:
                current_phase = traci.trafficlight.getPhase(TL_ID)
                if current_phase >= 0:
                    new_phase = (current_phase + 1) % 4  # koristi broj faza iz tvoje mreže
                    print(f"Changing phase from {current_phase} to {new_phase}")
                    traci.trafficlight.setPhase(TL_ID, new_phase)
                else:
                    print(f"Warning: Current phase is {current_phase}, skipping phase change.")
            last_action_time = step
        else:
            action = 0

        reward = -sum(state)
        total_reward += reward

        next_state = get_state()
        agent.learn(state, action, reward, next_state)
        state = next_state

    traci.close()
    return total_reward

# Main training loop
for ep in range(NUM_EPISODES):
    print(f"Starting episode {ep}")
    reward = run_episode(ep)

    with open(f"q-tabele-i-logovi/qtable_ep{ep}.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    with open("q-tabele-i-logovi/log.csv", "a") as log_file:
        log_file.write(f"{ep},{reward}\n")
    print(f"Episode {ep} total reward: {reward}\n")
