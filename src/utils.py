import os
import sys
import traci
import subprocess
from config import TL_ID, CONFIG_FILE, SUMO_BINARY_EVAL, MAX_STEPS, NUM_EPISODES, NUM_ROUTE_VARIATIONS, ALPHA, GAMMA, EPSILON, Q_TABLE_PATH, LAST_ALPHA, LAST_GAMMA, LAST_EPSILON, MIN_PHASE_DURATION, MAX_PHASE_DURATION, SUMO_BINARY

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        raise EnvironmentError("SUMO_HOME environment variable not set. Please add it to your shell config.")

def get_state():
    q_north = traci.lane.getLastStepVehicleNumber("e2_0")
    q_south = traci.lane.getLastStepVehicleNumber("e4_0")
    q_east  = traci.lane.getLastStepVehicleNumber("e6_0")
    q_west  = traci.lane.getLastStepVehicleNumber("e0_0")
    return (q_north, q_south, q_east, q_west)

def generate_random_routes(seed=None):
    command = [
        "python", 
        f"{os.environ['SUMO_HOME']}/tools/randomTrips.py",
        "-n", "network.net.xml",
        "-r", "routes.rou.xml",
        "-b", "0",
        "-e", "100",
        "-p", "1",
        "--validate"
    ]
    
    # Dodaj seed ako je proslijeđen
    if seed is not None:
        command.extend(["--seed", str(seed)])
    
    subprocess.run(command)

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
    min_phase_duration=MIN_PHASE_DURATION,
    max_phase_duration=MAX_PHASE_DURATION,
    q_table_path=Q_TABLE_PATH,
    last_alpha=LAST_ALPHA,
    last_gamma=LAST_GAMMA,
    last_epsilon=LAST_EPSILON,
    max_steps=MAX_STEPS,
    num_episodes=NUM_EPISODES,    
    num_route_variations=NUM_ROUTE_VARIATIONS
    ):
    with open("./config.py", "w") as config_file:
        config_file.write(f"""TL_ID = "{tl_id}"\n""")
        config_file.write(f"""MIN_PHASE_DURATION = {min_phase_duration}\n""")
        config_file.write(f"""MAX_PHASE_DURATION = {max_phase_duration}\n""")
        config_file.write(f"""CONFIG_FILE = "{sim_config_file}"\n""")
        config_file.write(f"""SUMO_BINARY = "{sumo_binary}"\n""")
        config_file.write(f"""SUMO_BINARY_EVAL = "{sumo_binary_eval}"\n""")
        config_file.write(f"""MAX_STEPS = {max_steps}\n""")
        config_file.write(f"""NUM_EPISODES = {num_episodes}\n""")
        config_file.write(f"""NUM_ROUTE_VARIATIONS = {num_route_variations}\n""")
        config_file.write(f"""ALPHA = {alpha}\n""")
        config_file.write(f"""GAMMA = {gamma}\n""")
        config_file.write(f"""EPSILON = {epsilon}\n""")
        config_file.write(f"""Q_TABLE_PATH = "{q_table_path}"\n""")
        config_file.write(f"""LAST_ALPHA = {last_alpha}\n""")
        config_file.write(f"""LAST_GAMMA = {last_gamma}\n""")
        config_file.write(f"""LAST_EPSILON = {last_epsilon}\n""")