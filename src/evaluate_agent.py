# evaluate_agent.py
import os
import pickle
import random
import traci
from utils import (
    QLearningAgent,
    check_sumo_home,
    generate_random_routes,
    get_phase_count,
    get_state,
    update_config,
)
from config import (
    NUM_ROUTE_VARIATIONS,
    SIMULATION_FOLDER,
    TL_ID,
    MIN_PHASE_DURATION,
    MAX_PHASE_DURATION,
    CONFIG_FILE,
    SUMO_BINARY_EVAL,
    MAX_STEPS,
)

check_sumo_home()

# Load learned Q-table
with open("q-tables-and-logs/qtable_final.pkl", "rb") as f:
    q_table = pickle.load(f)

agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table
agent.alpha = 0.0  # Disable learning during evaluation
agent.gamma = 0.9  # Keep the same gamma value as during training
agent.epsilon = 0.0  # Disable exploration during evaluation
# Start SUMO simulation


def evaluate_simulation(use_agent=True):
    traci.start([SUMO_BINARY_EVAL, "-c", CONFIG_FILE])
    step = 0
    last_action_time = 0
    departed_vehicles_number = 0
    arrived_vehicles_number = 0

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1
        departed_vehicles_number += traci.simulation.getDepartedNumber()
        arrived_vehicles_number += traci.simulation.getArrivedNumber()

        if (
            departed_vehicles_number == arrived_vehicles_number
            and departed_vehicles_number > 0
        ):
            eval_path = os.path.join("evaluation-results", "evaluation_results.csv")
            with open(eval_path, "a") as f:
                if use_agent:
                    f.write(f"{step}\n")
                else:
                    f.write(f"{step},")
            break

        state = get_state()

        if not use_agent:
            pass
        else:
            if step - last_action_time >= MIN_PHASE_DURATION:
                if step - last_action_time >= MAX_PHASE_DURATION:
                    action = 1
                else:
                    action = agent.choose_action(state)
                if action == 1:
                    current_phase = traci.trafficlight.getPhase(TL_ID)
                    if current_phase >= 0:
                        new_phase = (current_phase + 1) % get_phase_count()
                        print(f"Changing phase from {current_phase} to {new_phase}")
                        traci.trafficlight.setPhase(TL_ID, new_phase)
                    else:
                        print(
                            f"Warning: Current phase is {current_phase}, skipping phase change."
                        )
                    last_action_time = step

    traci.close()


if __name__ == "__main__":
    update_config(
        sim_end_of_generating=random.randint(200, 1700),
        routes_per_sec=random.random() * random.randint(4, 10) + random.random()
    )
    
    if os.path.exists(SIMULATION_FOLDER):
        os.chdir(SIMULATION_FOLDER)
        # use random routes for evaluation with seed that is in range of variations
        seed = random.randint(0, 1000) % NUM_ROUTE_VARIATIONS
        generate_random_routes(seed=seed)
        os.chdir("../src")

    print("Evaluating simulation without agent...")
    evaluate_simulation(use_agent=False)
    print("Finished evaluation without agent.\n")

    print("Evaluating simulation with agent...")
    evaluate_simulation(use_agent=True)
    print("Finished evaluation with agent.")
