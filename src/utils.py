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

# Test helpers
def _test_get_state_key():
    agent = QLearningAgent(actions=[0, 1])
    # Test with various queue lengths and durations
    state = (0, 30, 5, 10, 15, 20)
    key = agent.get_state_key(state)
    assert isinstance(key, tuple), "Key should be a tuple"
    assert len(key) == 6, "Key should have 6 elements (phase, duration_bin, 4 queue bins)"
    # Phase 0, duration 30 -> bin 3, queues: 5//5=1, 10//5=2, 15//5=3, 20//5=4
    assert key == (0, 3, 1, 2, 3, 4), f"Unexpected key: {key}"
    
    # Test with large queue values (clamping)
    state = (1, 100, 55, 60, 70, 5)
    key = agent.get_state_key(state)
    # Phase 1, duration 100 -> bin 10, queues: min(55,60)//5=11, min(60,60)//5=12, min(70,60)//5=12, 5//5=1
    assert key == (1, 10, 11, 12, 12, 1), f"Unexpected key: {key}"
    
    # Test with zero queues
    state = (2, 0, 0, 0, 0, 0)
    key = agent.get_state_key(state)
    assert key == (2, 0, 0, 0, 0, 0), f"Unexpected key: {key}"
    
    print("All _test_get_state_key tests passed!")

def _test_get_Q():
    agent = QLearningAgent(actions=[0, 1])
    # Initially all Q values should be 0.0
    state = (0, 10, 5, 5, 5, 5)
    assert agent.get_Q(state, 0) == 0.0
    assert agent.get_Q(state, 1) == 0.0
    
    # Set a Q value manually
    key = (agent.get_state_key(state), 0)
    agent.q_table[key] = 1.5
    assert agent.get_Q(state, 0) == 1.5
    assert agent.get_Q(state, 1) == 0.0  # Other action still 0
    
    print("All _test_get_Q tests passed!")

def _test_choose_action():
    agent = QLearningAgent(actions=[0, 1], epsilon=0.0)  # No exploration
    state = (0, 10, 5, 5, 5, 5)
    
    # With epsilon=0, should always choose best action
    # Initially all Q are 0, so should pick randomly among equal values
    action = agent.choose_action(state)
    assert action in [0, 1], f"Unexpected action: {action}"
    
    # Set Q values so action 1 is better
    key0 = (agent.get_state_key(state), 0)
    key1 = (agent.get_state_key(state), 1)
    agent.q_table[key0] = 1.0
    agent.q_table[key1] = 2.0
    
    # Should always pick action 1
    for _ in range(10):
        assert agent.choose_action(state) == 1, "Should always pick action 1"
    
    # Test with epsilon=1.0 (always random)
    agent.epsilon = 1.0
    actions_seen = set()
    for _ in range(100):
        actions_seen.add(agent.choose_action(state))
    assert actions_seen == {0, 1}, "Should see both actions with epsilon=1.0"
    
    print("All _test_choose_action tests passed!")

def _test_learn():
    agent = QLearningAgent(actions=[0, 1], alpha=0.5, gamma=0.9, epsilon=0.0)
    state = (0, 10, 5, 5, 5, 5)
    next_state = (1, 10, 3, 3, 3, 3)
    
    # Learn with reward 10
    agent.learn(state, 0, 10, next_state)
    key = (agent.get_state_key(state), 0)
    # new_q = 0 + 0.5 * (10 + 0.9 * 0 - 0) = 5.0
    assert agent.q_table[key] == 5.0, f"Expected 5.0, got {agent.q_table[key]}"
    
    # Learn again with same state-action but different next state
    agent.learn(state, 0, 5, next_state)
    # new_q = 5.0 + 0.5 * (5 + 0.9 * 0 - 5.0) = 5.0
    assert agent.q_table[key] == 5.0, f"Expected 5.0, got {agent.q_table[key]}"
    
    # Test with non-zero next_max_q
    next_key = (agent.get_state_key(next_state), 1)
    agent.q_table[next_key] = 3.0
    agent.learn(state, 0, 10, next_state)
    # new_q = 5.0 + 0.5 * (10 + 0.9 * 3.0 - 5.0) = 5.0 + 0.5 * (10 + 2.7 - 5.0) = 5.0 + 0.5 * 7.7 = 8.85
    assert abs(agent.q_table[key] - 8.85) < 1e-10, f"Expected 8.85, got {agent.q_table[key]}"
    
    print("All _test_learn tests passed!")

def _test_calculate_reward():
    # Test with small queues
    state = (0, 10, 1, 2, 1, 2)
    reward = calculate_reward(state)
    # queue_penalty = (1+4+1+4)/4 = 2.5, duration_penalty = 0
    assert reward == -2.5, f"Expected -2.5, got {reward}"
    
    # Test with large queues and long duration
    state = (0, 70, 25, 30, 20, 15)
    reward = calculate_reward(state)
    # queue_penalty = (625+900+400+225)/4 = 537.5
    # duration_penalty = (70-60)*1.2 = 12.0
    assert reward == -(537.5 + 12.0), f"Expected -549.5, got {reward}"
    
    # Test with large queues but short duration (no duration penalty)
    state = (0, 30, 25, 30, 20, 15)
    reward = calculate_reward(state)
    # queue_penalty = (625+900+400+225)/4 = 537.5, duration_penalty = 0
    assert reward == -537.5, f"Expected -537.5, got {reward}"
    
    # Test with max(queue) <= 20 but duration > 60
    state = (0, 70, 5, 10, 15, 20)
    reward = calculate_reward(state)
    # queue_penalty = (25+100+225+400)/4 = 187.5, duration_penalty = 0 (max queue <= 20)
    assert reward == -187.5, f"Expected -187.5, got {reward}"
    
    print("All _test_calculate_reward tests passed!")

def _test_get_state_key():
    _test_get_state_key()

def _test_get_Q():
    _test_get_Q()

def _test_choose_action():
    _test_choose_action()

def _test_learn():
    _test_learn()

def _test_calculate_reward():
    _test_calculate_reward()

if __name__ == "__main__":
    _test_get_state_key()
    _test_get_Q()
    _test_choose_action()
    _test_learn()
    _test_calculate_reward()
    print("All tests passed!")

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