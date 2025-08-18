import os
import random
import sys
import traci
import subprocess
import numpy as np
from config import (
    TL_ID,
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
        
    def get_state_key(self, state):
        """Optimizovana diskretizacija za velike protoke"""
        phase, duration, *queues = state
        
        MAX_QUEUE = 60 
        QUEUE_STEP = 5  
        
        queue_bins = [min(q, MAX_QUEUE) // QUEUE_STEP for q in queues]
        duration_bin = min(int(duration / 10), 10)
    
        return (phase, duration_bin) + tuple(queue_bins)

    def get_Q(self, state, action):
        key = (self.get_state_key(state), action)
        return self.q_table.get(key, 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = [self.get_Q(state, a) for a in self.actions]
        max_q = max(q_values)
        
        # Ako ima više akcija sa istom Q vrednošću
        max_indices = [i for i, q in enumerate(q_values) if q == max_q]
        return self.actions[random.choice(max_indices)]

    def learn(self, state, action, reward, next_state):
        current_key = (self.get_state_key(state), action)
        current_q = self.get_Q(state, action)
        
        # Max Q za sledeće stanje
        next_max_q = max([self.get_Q(next_state, a) for a in self.actions])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[current_key] = new_q

def check_sumo_home():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
        return True
    else:
        raise EnvironmentError("SUMO_HOME nije postavljen!")

def get_state(tls_id=TL_ID, conn=None):
    if conn is None:
        conn = traci
        
    try:
        current_phase = conn.trafficlight.getPhase(tls_id)
        phase_duration = conn.trafficlight.getPhaseDuration(tls_id)
        
        approaches = {}
        lane_counts = {}  # broj traka po prilazu
        
        for lane in conn.trafficlight.getControlledLanes(tls_id):
            edge_id = lane.split('_')[0]
            if edge_id not in approaches:
                approaches[edge_id] = 0
                lane_counts[edge_id] = 0
            approaches[edge_id] += conn.lane.getLastStepVehicleNumber(lane)
            lane_counts[edge_id] += 1
        
        # Pretvori u prosjek po traci u tom smjeru
        for edge_id in approaches:
            approaches[edge_id] /= lane_counts[edge_id]
        
        sorted_approaches = sorted(approaches.items())
        queue_lengths = [q for _, q in sorted_approaches]
        
        return (current_phase, phase_duration) + tuple(queue_lengths)
        
    except Exception as e:
        print(f"Greška u get_state: {e}")
        return (0, 0, 0, 0, 0, 0)


def calculate_reward(state):
    phase, duration, *queues = state
    
    # Penalizacija po prosječnom redu (kvadratna)
    queue_penalty = sum(q**2 for q in queues) / len(queues)  # normalizacija
    
    # Kazna za preduge faze (kada ima gužvi)
    if max(queues) > 20 and duration > 60:
        duration_penalty = (duration - 60) * 1.2
    else:
        duration_penalty = 0
        
    return -(queue_penalty + duration_penalty)

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
        "--validate",
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