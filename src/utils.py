import os
import random
import sys
import traci
import subprocess
import numpy as np
from config import (
    TL_ID,
    REWARD_CONFIG,
    SIM_START_OF_GENERATING,
    SIM_GENERATING_RANGE_MIN,
    SIM_GENERATING_RANGE_MAX,
    ROUTES_PER_SEC_RANGE_MIN,
    ROUTES_PER_SEC_RANGE_MAX,
    ROUTES_PER_SEC_RANGE_RANDOMIZE,
    NET_FILE,
    ROU_FILE
)

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.5):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_Q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = [self.get_Q(state, a) for a in self.actions]
        max_q = max(q_values)
        
        # Ako ima više akcija sa istom Q vrednošću
        max_indices = [i for i, q in enumerate(q_values) if q == max_q]
        return self.actions[random.choice(max_indices)]

    def learn(self, state, action, reward, next_state):
        current_q = self.get_Q(state, action)
        next_max_q = max([self.get_Q(next_state, a) for a in self.actions])
        
        # Q-learning formulu
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

def check_sumo_home():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
        return True
    else:
        raise EnvironmentError("SUMO_HOME nije postavljen!")

def get_state(tls_id=TL_ID):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        queue_lengths = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
        total_vehicles = sum(queue_lengths)
        max_queue = max(queue_lengths) if lanes else 0
        
        # Prosečno vreme čekanja
        waiting_times = []
        for lane in lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                waiting_times.append(traci.vehicle.getWaitingTime(veh_id))
        
        avg_waiting = np.mean(waiting_times) if waiting_times else 0
        
        # Trenutna faza i trajanje
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
        
        # Diskretizacija stanja (optimizovano)
        state_components = (
            min(total_vehicles // 5, 7),        # 0-7 (do 35 vozila)
            min(max_queue // 3, 7),              # 0-7 (do 21 vozila po traci)
            min(int(avg_waiting) // 10, 7),      # 0-7 (do 70 sekundi)
            current_phase,
            min(phase_duration // 10, 5)        # 0-5 (do 50 sekundi)
        )
        
        return state_components
        
    except traci.TraCIException:
        return (0, 0, 0, 0, 0)

def calculate_reward(tls_id, config):
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        
        # Kazna za gužvu
        queue_lengths = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
        queue_penalty = sum(queue_lengths) / config['queue_normalizer']
        
        # Kazna za čekanje
        waiting_times = []
        for lane in lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                waiting_times.append(traci.vehicle.getWaitingTime(veh_id))
        
        avg_waiting = np.mean(waiting_times) if waiting_times else 0
        waiting_penalty = avg_waiting / config['waiting_normalizer']
        
        flow_reward = traci.simulation.getArrivedNumber() / config['flow_normalizer']
        
        # Kombinovana nagrada
        reward = -(
            config['queue_weight'] * queue_penalty +
            config['waiting_weight'] * waiting_penalty
        ) + config['flow_weight'] * flow_reward
        
        return reward
        
    except traci.TraCIException:
        return 0

def generate_random_routes(seed=None):
    # Generisanje parametara
    sim_end = random.randint(SIM_GENERATING_RANGE_MIN, SIM_GENERATING_RANGE_MAX)
    routes_per_sec = random.uniform(ROUTES_PER_SEC_RANGE_MIN, ROUTES_PER_SEC_RANGE_MAX)
    
    if ROUTES_PER_SEC_RANGE_RANDOMIZE:
        routes_per_sec = round(routes_per_sec * random.uniform(0.8, 1.2), 2)
    
    # Komanda za generisanje ruta
    command = [
        "python", f"{os.environ['SUMO_HOME']}/tools/randomTrips.py",
        "-n", NET_FILE,
        "-r", ROU_FILE,
        "-b", str(SIM_START_OF_GENERATING),
        "-e", str(sim_end),
        "-p", str(1/routes_per_sec),
        "--validate"
    ]
    
    if seed is not None:
        command.extend(["--seed", str(seed)])
    
    # Pokretanje procesa
    try:
        subprocess.run(command, check=True)
        return sim_end
    except subprocess.CalledProcessError as e:
        print(f"Greška pri generisanju ruta: {e}")
        return SIM_GENERATING_RANGE_MAX

def get_phase_count(tls_id=TL_ID):
    try:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        return len(program.getPhases())
    except (traci.TraCIException, IndexError):
        return 4  # Podrazumevana vrednost

def update_config(**kwargs):
    config_lines = []
    with open("./config.py", "r") as config_file:
        for line in config_file:
            if any(line.startswith(f"{key} =") for key in kwargs):
                continue
            config_lines.append(line)
    
    with open("./config.py", "w") as config_file:
        config_file.writelines(config_lines)
        for key, value in kwargs.items():
            # Formatiranje floating-point vrijednosti na 6 decimala
            if isinstance(value, float):
                config_file.write(f'{key} = {value:.6f}\n')
            elif isinstance(value, str):
                config_file.write(f'{key} = "{value}"\n')
            else:
                config_file.write(f'{key} = {value}\n')