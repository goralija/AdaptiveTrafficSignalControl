import os
import random
import sys
import traci
import subprocess
from config import (
    EPISODES_DONE,
    NET_FILE,
    ROU_FILE,
    SIMULATION_FOLDER,
    TL_ID,
    CONFIG_FILE,
    SUMO_BINARY_EVAL,
    MAX_STEPS,
    NUM_EPISODES,
    NUM_ROUTE_VARIATIONS,
    ALPHA,
    GAMMA,
    EPSILON,
    Q_TABLE_PATH,
    LAST_ALPHA,
    LAST_GAMMA,
    LAST_EPSILON,
    MIN_PHASE_DURATION,
    MAX_PHASE_DURATION,
    SUMO_BINARY,
    NUM_EVAL_EPISODES,
    ALPHA_DECAY,
    EPSILON_DECAY,
    ROUTES_PER_SEC,
    SIM_START_OF_GENERATING,
    SIM_END_OF_GENERATING,
    SIM_GENERATING_RANGE_MAX,
    SIM_GENERATING_RANGE_MIN,
    ROUTES_PER_SEC_RANGE_MIN,
    ROUTES_PER_SEC_RANGE_MAX,
    ROUTES_PER_SEC_RANGE_RANDOMIZE
)


class QLearningAgent:
    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.q_table = {}  # key: (state, action), value: Q-value
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
        return self.actions[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state):
        current_q = self.get_Q(state, action)
        max_future_q = max([self.get_Q(next_state, a) for a in self.actions])
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )
        self.q_table[(state, action)] = new_q


def check_sumo_home():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        raise EnvironmentError(
            "SUMO_HOME environment variable not set. Please add it to your shell config."
        )


def get_state(tls_id=TL_ID):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    # Keep unique lanes only
    lanes = list(set(lanes))
    queue_lengths = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
    return tuple(queue_lengths)

def generate_random_routes(seed=None):
    os.chdir('../src/')
    sim_end_of_generating = random.randint(SIM_GENERATING_RANGE_MIN, SIM_GENERATING_RANGE_MAX)
    routes_per_sec = random.randint(ROUTES_PER_SEC_RANGE_MIN, ROUTES_PER_SEC_RANGE_MAX)
    routes_per_sec = random.random() * routes_per_sec + random.random() if ROUTES_PER_SEC_RANGE_RANDOMIZE else routes_per_sec
    update_config(sim_end_of_generating=sim_end_of_generating, routes_per_sec=routes_per_sec)
    os.chdir('../simulation-config/')
    command = [
        "python",
        f"{os.environ['SUMO_HOME']}/tools/randomTrips.py",
        "-n",
        f"{NET_FILE}",
        "-r",
        f"{ROU_FILE}",
        "-b",
        str(SIM_START_OF_GENERATING),
        "-e",
        str(sim_end_of_generating),
        "-p",
        str(routes_per_sec),
        "--validate",
    ]

    # Dodaj seed ako je proslijeđen
    if seed is not None:
        command.extend(["--seed", str(seed)])

    subprocess.run(command)
    return sim_end_of_generating


def get_phase_count():
    num_phases = traci.trafficlight.getAllProgramLogics(TL_ID)[0].getPhases()
    return len(num_phases)


def update_config(
    sumo_binary=SUMO_BINARY,
    sumo_binary_eval=SUMO_BINARY_EVAL,
    sim_config_file=CONFIG_FILE,
    tl_id=TL_ID,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    alpha_decay=ALPHA_DECAY,
    epsilon_decay=EPSILON_DECAY,
    min_phase_duration=MIN_PHASE_DURATION,
    max_phase_duration=MAX_PHASE_DURATION,
    q_table_path=Q_TABLE_PATH,
    last_alpha=LAST_ALPHA,
    last_gamma=LAST_GAMMA,
    last_epsilon=LAST_EPSILON,
    max_steps=MAX_STEPS,
    episodes_done=EPISODES_DONE,
    num_episodes=NUM_EPISODES,
    num_eval_episodes=NUM_EVAL_EPISODES,
    num_route_variations=NUM_ROUTE_VARIATIONS,
    simulation_folder=SIMULATION_FOLDER,
    net_file=NET_FILE,
    rou_file=ROU_FILE,
    routes_per_sec=ROUTES_PER_SEC,
    sim_start_of_generating=SIM_START_OF_GENERATING,
    sim_end_of_generating=SIM_END_OF_GENERATING,
    sim_generating_range_min=SIM_GENERATING_RANGE_MIN,
    sim_generating_range_max=SIM_GENERATING_RANGE_MAX,
    routes_per_sec_range_min=ROUTES_PER_SEC_RANGE_MIN,
    routes_per_sec_range_max=ROUTES_PER_SEC_RANGE_MAX,
    routes_per_sec_range_randomize=ROUTES_PER_SEC_RANGE_RANDOMIZE
):
    with open("./config.py", "w") as config_file:
        config_file.write(f"""TL_ID = "{tl_id}"\n""")
        config_file.write(f"""MIN_PHASE_DURATION = {min_phase_duration}\n""")
        config_file.write(f"""MAX_PHASE_DURATION = {max_phase_duration}\n""")
        config_file.write(f"""CONFIG_FILE = "{sim_config_file}"\n""")
        config_file.write(f"""SIMULATION_FOLDER = "{simulation_folder}"\n""")
        config_file.write(f"""SIM_GENERATING_RANGE_MIN = {sim_generating_range_min}\n""")
        config_file.write(f"""SIM_GENERATING_RANGE_MAX = {sim_generating_range_max}\n""")
        config_file.write(f"""SIM_START_OF_GENERATING = {sim_start_of_generating}\n""")
        config_file.write(f"""SIM_END_OF_GENERATING = {sim_end_of_generating}\n""")
        config_file.write(f"""ROUTES_PER_SEC_RANGE_MIN = {routes_per_sec_range_min}\n""")
        config_file.write(f"""ROUTES_PER_SEC_RANGE_MAX = {routes_per_sec_range_max}\n""")
        config_file.write(f"""ROUTES_PER_SEC_RANGE_RANDOMIZE = {routes_per_sec_range_randomize}\n""")
        config_file.write(f"""ROUTES_PER_SEC = {routes_per_sec}\n""")
        config_file.write(f"""NET_FILE = "{net_file}"\n""")
        config_file.write(f"""ROU_FILE = "{rou_file}"\n""")
        config_file.write(f"""SUMO_BINARY = "{sumo_binary}"\n""")
        config_file.write(f"""SUMO_BINARY_EVAL = "{sumo_binary_eval}"\n""")
        config_file.write(f"""MAX_STEPS = {max_steps}\n""")
        config_file.write(f"""EPISODES_DONE = {episodes_done}\n""")
        config_file.write(f"""NUM_EPISODES = {num_episodes}\n""")
        config_file.write(f"""NUM_EVAL_EPISODES = {num_eval_episodes}\n""")
        config_file.write(f"""NUM_ROUTE_VARIATIONS = {num_route_variations}\n""")
        config_file.write(f"""ALPHA = {alpha}\n""")
        config_file.write(f"""GAMMA = {gamma}\n""")
        config_file.write(f"""EPSILON = {epsilon}\n""")
        config_file.write(f"""ALPHA_DECAY = {alpha_decay}\n""")
        config_file.write(f"""EPSILON_DECAY = {epsilon_decay}\n""")
        config_file.write(f"""Q_TABLE_PATH = "{q_table_path}"\n""")
        config_file.write(f"""LAST_ALPHA = {last_alpha}\n""")
        config_file.write(f"""LAST_GAMMA = {last_gamma}\n""")
        config_file.write(f"""LAST_EPSILON = {last_epsilon}\n""")
