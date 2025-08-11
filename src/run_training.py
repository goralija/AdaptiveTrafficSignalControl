import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import sys
import traci
from utils import (
    QLearningAgent,
    check_sumo_home,
    get_state,
    generate_random_routes,
    get_phase_count,
    update_config,
    calculate_reward  # Dodata nova funkcija za nagradu
)
from config import (
    ALPHA_DECAY,
    EPISODES_DONE,
    EPSILON_DECAY,
    TL_ID,
    NUM_ROUTE_VARIATIONS,
    MIN_PHASE_DURATION,
    MAX_PHASE_DURATION,
    CONFIG_FILE,
    SUMO_BINARY,
    MAX_STEPS,
    NUM_EPISODES,
    Q_TABLE_PATH,
    LAST_ALPHA,
    LAST_GAMMA,
    LAST_EPSILON,
    SIMULATION_FOLDER,
    REWARD_CONFIG  # Novi konfig objekat
)

check_sumo_home()

if "--new" in sys.argv:
    print("Cleaning previous training artifacts...")
    if os.path.exists("evaluation-results"):
        shutil.rmtree("evaluation-results")
    if os.path.exists("q-tables-and-logs"):
        shutil.rmtree("q-tables-and-logs")

if os.path.exists(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "rb") as f:
        loaded_q_table = pickle.load(f)
    print("Učitana postojeća Q-tabela!")
    agent = QLearningAgent(
        actions=[0, 1], alpha=LAST_ALPHA, gamma=LAST_GAMMA, epsilon=LAST_EPSILON
    )
    agent.q_table = loaded_q_table
else:
    agent = QLearningAgent(actions=[0, 1])
    print("Nema postojeće Q-tabele, kreiran novi agent!")
    try:
        if os.path.exists("q-tables-and-logs"):
            shutil.rmtree("q-tables-and-logs")
        os.makedirs("q-tables-and-logs")
        os.makedirs("q-tables-and-logs/tables", exist_ok=True)
    except Exception as e:
        print(f"Greška pri kreiranju direktorijuma: {e}")

def run_episode(episode, sim_folder=SIMULATION_FOLDER):
    if not os.path.exists(sim_folder):
        print(f"Direktorijum '{sim_folder}' ne postoji!")
        return (0, 0, 0, 0, 0)
    
    os.chdir(sim_folder)
    seed = episode % NUM_ROUTE_VARIATIONS
    sim_generating_end = generate_random_routes(seed)
    os.chdir("../src")
    
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    step = 0
    last_action_time = 0
    total_reward = 0
    departed_vehicles = 0
    arrived_vehicles = 0
    departures_ended = False
    cumulative_waiting = 0
    measurement_count = 0
    
    # Inicijalizacija stanja
    lanes = traci.trafficlight.getControlledLanes(TL_ID)
    state = get_state(TL_ID)

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1
        
        current_departed = traci.simulation.getDepartedNumber()
        current_arrived = traci.simulation.getArrivedNumber()
        
        if step >= sim_generating_end:
            departures_ended = True

        # Provjera završetka simulacije
        if departures_ended:
            if arrived_vehicles+current_arrived >= departed_vehicles+current_departed:
                break

        # Prikupljanje podataka o čekanju
        waiting_times = [traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList()]
        if waiting_times:
            cumulative_waiting += sum(waiting_times)
            measurement_count += len(waiting_times)

        # Izbor akcije
        current_phase = traci.trafficlight.getPhase(TL_ID)
        if current_phase == -1:
            continue

        if step - last_action_time >= MIN_PHASE_DURATION:
            if step - last_action_time >= MAX_PHASE_DURATION:
                action = 1
            else:
                action = agent.choose_action(state)
                
            if action == 1:
                new_phase = (current_phase + 1) % get_phase_count()
                traci.trafficlight.setPhase(TL_ID, new_phase)
                last_action_time = step
        else:
            action = 0

        # Izračun nagrade
        reward = calculate_reward(TL_ID, REWARD_CONFIG)
        total_reward += reward

        # Učenje agenta
        next_state = get_state(TL_ID)
        agent.learn(state, action, reward, next_state)
        state = next_state
        
        # update pokrenutih i pristiglih vozila
        departed_vehicles += current_departed
        arrived_vehicles += current_arrived

    traci.close()
    
    # Izračun prosečnog vremena čekanja
    avg_waiting = cumulative_waiting / measurement_count if measurement_count > 0 else 0
    
    return (total_reward, step, sim_generating_end, arrived_vehicles, avg_waiting)

# Glavna petlja treniranja
for ep in range(EPISODES_DONE + 1, NUM_EPISODES + 1):
    agent.epsilon *= EPSILON_DECAY
    agent.alpha *= ALPHA_DECAY
    
    print(f"Početak epizode {ep}")
    reward, steps, gen_end, arrived, avg_wait = run_episode(ep)
    
    # Čuvanje Q-tabele
    if ep % 50 == 0 or ep == NUM_EPISODES:
        table_path = f"q-tables-and-logs/tables/qtable_ep{ep}.pkl"
        with open(table_path, "wb") as f:
            pickle.dump(agent.q_table, f)
        print(f"Sačuvana Q-tabela: {table_path}")
    
    # Logovanje rezultata
    log_entry = f"{ep},{reward},{gen_end},{steps},{arrived},{avg_wait}\n"
    with open("q-tables-and-logs/log.csv", "a") as log_file:
        if ep == 1:
            log_file.write("Episode,Total Reward,Gen End,Sim End,Arrived Vehicles,Avg Waiting\n")
        log_file.write(log_entry)
    
    print(f"Epizoda {ep} završena: Nagrada={reward:.2f}, Vozila={arrived}, Čekanje={avg_wait:.2f}s")
    
    # Ažuriranje konfiguracije
    update_config(
        last_alpha=round(agent.alpha, 6),
        last_gamma=round(agent.gamma, 6),
        last_epsilon=round(agent.epsilon, 6),
        episodes_done=ep
    )

# Generisanje grafikona
if os.path.exists("q-tables-and-logs/log.csv"):
    os.chdir("q-tables-and-logs")
    df = pd.read_csv("log.csv")
    
    plt.figure(figsize=(12, 6))
    
    # Grafikon nagrada
    plt.subplot(1, 2, 1)
    df['Smoothed Reward'] = df['Total Reward'].rolling(window=50).mean()
    plt.plot(df['Episode'], df['Smoothed Reward'], 'b-')
    plt.xlabel('Epizoda')
    plt.ylabel('Prosečna nagrada')
    plt.title('Tok treniranja')
    plt.grid(True)
    
    # Grafikon vremena čekanja
    plt.subplot(1, 2, 2)
    df['Smoothed Waiting'] = df['Avg Waiting'].rolling(window=50).mean()
    plt.plot(df['Episode'], df['Smoothed Waiting'], 'r-')
    plt.xlabel('Epizoda')
    plt.ylabel('Prosečno čekanje (s)')
    plt.title('Vreme čekanja vozila')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    print("Grafikoni sačuvani kao training_progress.png")
    os.chdir("..")
else:
    print("Log fajl nije pronađen, preskačem generisanje grafikona")