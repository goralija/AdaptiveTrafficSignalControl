# evaluate_comparison.py
import traci
import pandas as pd
from utils import get_state, generate_random_routes
from config import (
    TL_ID, MIN_PHASE_DURATION, MAX_PHASE_DURATION,
    CONFIG_FILE, SUMO_BINARY, SUMO_BINARY_EVAL, MAX_STEPS, NUM_EVAL_EPISODES
)
from qlearning_agent import QLearningAgent
import pickle

# Load the trained RL agent
with open('q-tables-and-logs/qtable_final.pkl', 'rb') as f:
    q_table = pickle.load(f)
rl_agent = QLearningAgent(actions=[0, 1])
rl_agent.q_table = q_table
rl_agent.epsilon = 0  # Disable exploration during evaluation

# Fixed-cycle controller parameters
FIXED_CYCLE = [20, 30, 20, 30]  # Phase durations in steps (adjust based on your traffic light phases)

def run_simulation(use_rl_agent=True, episode=0):
    # Generate random routes for this evaluation episode
    seed = episode % NUM_EVAL_EPISODES + 23
    generate_random_routes(seed=seed)  # Modified to include episode-specific routes
    
    traci.start([SUMO_BINARY_EVAL if use_rl_agent else SUMO_BINARY, "-c", CONFIG_FILE])
    step = 0
    last_action_time = 0
    metrics = {
        "total_waiting_time": 0,
        "total_vehicles": 0,
        "avg_speed": 0
    }
    phase_changes = 0

    if use_rl_agent:
        state = get_state()
    else:
        current_phase = 0
        phase_step = 0

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        # Collect metrics (modify based on your needs)
        metrics["total_waiting_time"] += traci.edge.getWaitingTime("edge_id")  # Replace with your edge/lane IDs
        metrics["total_vehicles"] += traci.vehicle.getIDCount()
        metrics["avg_speed"] += traci.vehicle.getSpeed(traci.vehicle.getIDList()[0])  # Example

        # Phase control logic
        if use_rl_agent:
            # RL Agent logic
            if step - last_action_time >= MIN_PHASE_DURATION:
                if step - last_action_time >= MAX_PHASE_DURATION:
                    action = 1
                else:
                    action = rl_agent.choose_action(state)
                if action == 1:
                    current_phase = traci.trafficlight.getPhase(TL_ID)
                    new_phase = (current_phase + 1) % 4  # Adjust for your phase count
                    traci.trafficlight.setPhase(TL_ID, new_phase)
                    last_action_time = step
                    phase_changes += 1
                state = get_state()  # Update state after action
        else:
            # Fixed-cycle controller logic
            phase_step += 1
            if phase_step >= FIXED_CYCLE[current_phase]:
                current_phase = (current_phase + 1) % len(FIXED_CYCLE)
                traci.trafficlight.setPhase(TL_ID, current_phase)
                phase_step = 0
                phase_changes += 1

    traci.close()
    metrics["avg_speed"] = metrics["avg_speed"] / metrics["total_vehicles"] if metrics["total_vehicles"] > 0 else 0
    return metrics, phase_changes

# Run comparison for NUM_EVAL_EPISODES
results = []
for ep in range(NUM_EVAL_EPISODES):
    # Ensure both simulations use the SAME random routes for this episode
    rl_metrics, rl_phase_changes = run_simulation(use_rl_agent=True, episode=ep)
    fixed_metrics, fixed_phase_changes = run_simulation(use_rl_agent=False, episode=ep)
    
    results.append({
        "episode": ep,
        "rl_waiting_time": rl_metrics["total_waiting_time"],
        "fixed_waiting_time": fixed_metrics["total_waiting_time"],
        "rl_phase_changes": rl_phase_changes,
        "fixed_phase_changes": fixed_phase_changes,
    })

    # Save intermediate results
    pd.DataFrame(results).to_csv("comparison_results.csv", index=False)
    print(f"Episode {ep} completed. RL Waiting: {rl_metrics['total_waiting_time']}, Fixed: {fixed_metrics['total_waiting_time']}")