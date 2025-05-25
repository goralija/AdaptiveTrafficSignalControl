# run_training.py
import pickle
import os
import shutil
import traci
from qlearning_agent import QLearningAgent
from utils import check_sumo_home, get_state, generate_random_routes, get_phase_count, update_config
from config import TL_ID, NUM_ROUTE_VARIATIONS, MIN_PHASE_DURATION, MAX_PHASE_DURATION, CONFIG_FILE, SUMO_BINARY, MAX_STEPS, NUM_EPISODES, Q_TABLE_PATH, LAST_ALPHA, LAST_GAMMA, LAST_EPSILON

check_sumo_home()

if os.path.exists(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "rb") as f:
        loaded_q_table = pickle.load(f)
    print("Učitana postojeća Q-tabela!")
    agent = QLearningAgent(actions=[0, 1], alpha=LAST_ALPHA, gamma=LAST_GAMMA, epsilon=LAST_EPSILON)  # Koristi posljednje vrijednosti alpha, gamma i epsilon
    agent.q_table = loaded_q_table  # Koristi postojeću tabelu
else:
    agent = QLearningAgent(actions=[0, 1])  # Kreiraj novog agenta
    print("Nema postojeće Q-tabele, kreiran novi agent!")
# preparing the directory for Q-tables and logs
try:
    if os.path.exists("q-tables-and-logs"):
        shutil.rmtree("q-tables-and-logs")
    os.makedirs("q-tables-and-logs")
    os.chdir("q-tables-and-logs")
    os.makedirs("tables")
    os.chdir("..")
except FileExistsError:
    print("Directory already exists, skipping creation.")
except FileNotFoundError:
    print("Directory not found, creating a new one.")
    os.makedirs("q-tables-and-logs")
except PermissionError:
    print("Permission denied, unable to create directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
def run_episode(episode):
    if os.path.exists("../simulation"):
        os.chdir("../simulation")
        seed = episode % NUM_ROUTE_VARIATIONS
        generate_random_routes(seed=seed)
        os.chdir("../src")
    else:
        print("Directory '../simulation' does not exist, please check the path.")
        return
        
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    step = 0
    last_action_time = 0
    total_reward = 0

    state = get_state()

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        current_phase = traci.trafficlight.getPhase(TL_ID)
        #print(f"Step {step} - Current phase: {current_phase}")

        if current_phase == -1:
            # Semafor nije spreman, ne menjaj fazu i samo nastavi
            continue

        if step - last_action_time >= MIN_PHASE_DURATION:
            if step - last_action_time >= MAX_PHASE_DURATION:
                action = 1
            else:
                action = agent.choose_action(state)
            if action == 1:
                current_phase = traci.trafficlight.getPhase(TL_ID)
                if current_phase >= 0:
                    new_phase = (current_phase + 1) % get_phase_count()  # koristi broj faza iz tvoje mreže
                    print(f"Changing phase from {current_phase} to {new_phase} in step {step}")
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
    # adjust hyperparameters for each episode
    agent.epsilon = 0.1 * (0.97 ** ep)
    agent.alpha = 0.1 / (1 + ep * 0.001)
    
    print(f"Starting episode {ep}")
    reward = run_episode(ep)

    # save the Q-table after every nth episode
    if (ep+1) % NUM_ROUTE_VARIATIONS == 0:
        with open(f"q-tables-and-logs/tables/qtable_ep{ep}.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)
    
    if ep == NUM_EPISODES - 1:
        with open("q-tables-and-logs/qtable_final.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)

    with open("q-tables-and-logs/log.csv", "a") as log_file:
        log_file.write(f"{ep},{reward}\n")
    print(f"Episode {ep} total reward: {reward}\n")
    
    print (agent.alpha, agent.gamma, agent.epsilon)
    # change hyperparameters in config.py for the next training run
    update_config(last_alpha=agent.alpha, last_gamma=agent.gamma, last_epsilon=agent.epsilon)