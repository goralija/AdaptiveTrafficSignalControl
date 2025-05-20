import traci
import sumolib
import time
from qlearning_agent import QLearningAgent
from utils import check_sumo_home
import pickle


check_sumo_home()

# Configuration
SUMO_BINARY = "sumo"  # or "sumo-gui" for GUI
CONFIG_FILE = "simulation/config.sumocfg"
TL_ID = "TL"  # traffic light ID from your net.xml
PHASE_DURATION = 10  # minimum phase duration (s)

agent = QLearningAgent(actions=[0, 1])  # 0: keep phase, 1: change phase

# Training loop
for episode in range(100):
    print(f"Starting episode {episode}")
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    total_reward = 0
    step = 0
    last_action_time = 0

    state = None
    while step < 1000:
        traci.simulationStep()
        step += 1

        # Collect state
        q_north = traci.lane.getLastStepVehicleNumber("N2TL")
        q_south = traci.lane.getLastStepVehicleNumber("S2TL")
        q_east  = traci.lane.getLastStepVehicleNumber("E2TL")
        q_west  = traci.lane.getLastStepVehicleNumber("W2TL")
        t_phase = traci.trafficlight.getPhaseDuration(TL_ID)
        state = (q_north, q_south, q_east, q_west)

        # Choose and apply action only every PHASE_DURATION steps
        if step - last_action_time >= PHASE_DURATION:
            action = agent.choose_action(state)
            if action == 1:
                current_phase = traci.trafficlight.getPhase(TL_ID)
                new_phase = 1 - current_phase
                traci.trafficlight.setPhase(TL_ID, new_phase)
            last_action_time = step

        # Compute reward
        reward = - (q_north + q_south + q_east + q_west)
        total_reward += reward

        # Observe next state
        q_north_new = traci.lane.getLastStepVehicleNumber("N2TL")
        q_south_new = traci.lane.getLastStepVehicleNumber("S2TL")
        q_east_new  = traci.lane.getLastStepVehicleNumber("E2TL")
        q_west_new  = traci.lane.getLastStepVehicleNumber("W2TL")
        next_state = (q_north_new, q_south_new, q_east_new, q_west_new)

        # Learn
        agent.learn(state, action, reward, next_state)

    with open(f'qtable_ep{episode}.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
        
    print(f'Episode {episode} total reward: {total_reward}')
    with open("log.csv", "a") as log_file:
        log_file.write(f"{episode},{total_reward}\n")

    traci.close()
    print(f"Finished episode {episode}\n")
    
