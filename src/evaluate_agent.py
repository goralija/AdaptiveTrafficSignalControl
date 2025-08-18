import os
import pickle
import random
import traci
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    QLearningAgent,
    check_sumo_home,
    generate_random_routes,
    get_phase_count,
    get_state,
    calculate_reward
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
    NUM_EVAL_EPISODES,
    EVAL_Q_TABLE_PATH,
    episodes_done
)

check_sumo_home()

EVAL_RESULTS_DIR = os.path.join("evaluation-results", str(episodes_done+1))
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Load learned Q-table
try:
    with open(EVAL_Q_TABLE_PATH+str(episodes_done+1)+".pkl", "rb") as f:
        q_table = pickle.load(f)
    print("Uspješno učitana Q-tabela za evaluaciju!")
except FileNotFoundError:
    print("Greška: Q-tabela nije pronađena. Prvo izvršite treniranje.")
    exit(1)

# Initialize agent with learned Q-table
agent = QLearningAgent(actions=[0, 1])
agent.q_table = q_table
agent.alpha = 0.0    # Onemogući učenje tokom evaluacije
agent.epsilon = 0.0  # Onemogući istraživanje tokom evaluacije

def evaluate_simulation(use_agent=True, seed=None):
    """Pokreće jednu simulaciju i prikuplja metriku performansi"""
    # Generiši rute sa zadatim seed-om
    # os.chdir(SIMULATION_FOLDER)
    # sim_generating_end = generate_random_routes(seed)
    # os.chdir("../src")
    
    # Pokreni SUMO
    traci.start([SUMO_BINARY_EVAL, "-c", CONFIG_FILE, "--no-warnings", "--no-step-log"])
    
    step = 0
    last_action_time = 0
    cumulative_waiting = 0
    measurement_count = 0
    total_reward = 0
    total_departed = 0
    total_arrived = 0
    cumulative_queue_length = 0
    queue_measurement_count = 0
    
    # Metrike za praćenje
    metrics = {
        'total_steps': 0,
        'departed': 0,
        'arrived': 0,
        'avg_waiting': 0,
        'total_reward': 0
    }
    
    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1
        
        # Ažuriraj broj vozila
        departed = traci.simulation.getDepartedNumber()
        arrived = traci.simulation.getArrivedNumber()
        
        # Prikupi podatke o čekanju
        waiting_times = [traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList()]
        if waiting_times:
            cumulative_waiting += sum(waiting_times)
            measurement_count += len(waiting_times)
        
        # Provjera kraja simulacije
        if step > sim_generating_end:
            if total_arrived+arrived >= total_departed+departed:
                break
            
        current_state = get_state(TL_ID)
        total_queue = sum(current_state[2:])  # sve nakon faze i trajanja su redovi
        cumulative_queue_length += total_queue
        queue_measurement_count += 1

        
        # Ako koristimo agenta, odredi akciju
        if use_agent:
            current_phase = traci.trafficlight.getPhase(TL_ID)
            if current_phase == -1:
                continue

            current_state = get_state(TL_ID)
                
            if step - last_action_time >= MIN_PHASE_DURATION:
                if step - last_action_time >= MAX_PHASE_DURATION:
                    action = 1
                else:
                    action = agent.choose_action(current_state)
                
                if action == 1:
                    #current_phase = traci.trafficlight.getPhase(TL_ID)
                    new_phase = (current_phase + 1) % get_phase_count(TL_ID)
                    traci.trafficlight.setPhase(TL_ID, new_phase)
                    last_action_time = step
                    #phase_options = list(range(get_phase_count()))
                    #phase_options.remove(current_phase)  # Ukloni trenutnu fazu
                    #
                    ## Pronađi najbolju fazu na osnovu Q-vrijednosti
                    #best_phase = current_phase
                    #best_value = float('-inf')
                    #
                    #for phase in phase_options:
                    #    # Hipotetičko stanje za ovu fazu
                    #    hyp_state = (current_state[0], current_state[1], phase) + current_state[3:]
                    #    state_value = max([agent.get_Q(hyp_state, a) for a in agent.actions])
                    #    
                    #    if state_value > best_value:
                    #        best_value = state_value
                    #        best_phase = phase
                    #
                    ## Primijeni promjenu
                    #traci.trafficlight.setPhase(TL_ID, best_phase)
                    #last_action_time = step
            else:
                action = 0
            
            # Izračunaj nagradu (samo za praćenje, ne za učenje)
            reward = calculate_reward(current_state)
            total_reward += reward        
            
        total_arrived += arrived
        total_departed += departed
    
    # Prikupi finalne metrike
    metrics['total_steps'] = step
    metrics['departed'] = departed
    metrics['arrived'] = arrived
    metrics['avg_waiting'] = cumulative_waiting / measurement_count if measurement_count > 0 else 0
    metrics['total_reward'] = total_reward
    metrics['avg_queue_length'] = cumulative_queue_length / queue_measurement_count if queue_measurement_count > 0 else 0

    
    traci.close()
    return metrics

def save_results(results, filename="evaluation_results.csv"):
    """Čuva rezultate evaluacije u CSV fajl"""
    os.makedirs("evaluation-results", exist_ok=True)
    path = os.path.join(EVAL_RESULTS_DIR, filename)
    
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Rezultati sačuvani u {path}")
    
    return df

def plot_results(df):
    """Generiše grafikone za rezultate evaluacije"""
    plt.figure(figsize=(15, 10))
    
    # Grafikoni za vreme čekanja
    plt.subplot(2, 2, 1)
    plt.bar(df['run'], df['agent_avg_waiting'], color='blue', alpha=0.7, label='Sa agentom')
    plt.bar(df['run'], df['fixed_avg_waiting'], color='red', alpha=0.7, label='Fiksni vremena')
    plt.xlabel('Pokretanje')
    plt.ylabel('Prosečno vreme čekanja (s)')
    plt.title('Upoređenje vremena čekanja')
    plt.legend()
    plt.grid(True)
    
    # Grafikoni za broj koraka
    plt.subplot(2, 2, 2)
    plt.plot(df['run'], df['agent_steps'], 'b-o', label='Sa agentom')
    plt.plot(df['run'], df['fixed_steps'], 'r-o', label='Fiksni vremena')
    plt.xlabel('Pokretanje')
    plt.ylabel('Ukupno koraka simulacije')
    plt.title('Trajanje simulacije')
    plt.legend()
    plt.grid(True)
    
    # Grafikoni za nagradu
    plt.subplot(2, 2, 3)
    plt.bar(df['run'], df['agent_reward'], color='green', alpha=0.7)
    plt.xlabel('Pokretanje')
    plt.ylabel('Ukupna nagrada')
    plt.title('Nagrada agenta po pokretanju')
    plt.grid(True)
    
    # Grafikoni za efikasnost
    plt.subplot(2, 2, 4)
    efficiency = (df['fixed_steps'] - df['agent_steps']) / df['fixed_steps'] * 100
    plt.bar(df['run'], efficiency, color='purple', alpha=0.7)
    plt.xlabel('Pokretanje')
    plt.ylabel('Poboljšanje (%)')
    plt.title('Efikasnost agenta u odnosu na fiksna vremena')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_RESULTS_DIR, "evaluation_comparison.png"))
    print(f"Grafikoni sačuvani kao {os.path.join(EVAL_RESULTS_DIR, 'evaluation_comparison.png')}")

def plot_queue_lengths(df):
    plt.figure(figsize=(8, 6))
    plt.bar(df['run'] - 0.2, df['fixed_avg_queue'], width=0.4, label='Fiksni ciklusi', alpha=0.7, color='red')
    plt.bar(df['run'] + 0.2, df['agent_avg_queue'], width=0.4, label='Agent', alpha=0.7, color='blue')
    plt.xlabel('Pokretanje')
    plt.ylabel('Prosječna dužina reda (vozila)')
    plt.title('Upoređenje prosječne dužine redova čekanja')
    plt.legend()
    plt.grid(True)
    file_path = os.path.join(EVAL_RESULTS_DIR, "queue_length_comparison.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Grafik reda čekanja sačuvan kao {file_path}")

if __name__ == "__main__":
    results = []
    
    for i in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n{'='*50}")
        print(f"Evaluacijsko pokretanje {i}/{NUM_EVAL_EPISODES}")
        
        # Generiši jedinstveni seed za obe varijante
        seed = i % NUM_ROUTE_VARIATIONS
        
        os.chdir(SIMULATION_FOLDER)
        sim_generating_end = generate_random_routes(seed)
        os.chdir("../src")
        
        # Pokreni sa fiksnim vremenima semafora
        print("Pokrećem simulaciju sa FIKSNIM vremenima semafora...")
        fixed_metrics = evaluate_simulation(use_agent=False, seed=seed)
        print(f"Završeno sa fiksnim vremenima: Koraci={fixed_metrics['total_steps']}, Čekanje={fixed_metrics['avg_waiting']:.2f}s")
        
        # Pokreni sa agentom
        print("Pokrećem simulaciju sa AGENTOM...")
        agent_metrics = evaluate_simulation(use_agent=True, seed=seed)
        print(f"Završeno sa agentom: Koraci={agent_metrics['total_steps']}, Čekanje={agent_metrics['avg_waiting']:.2f}s, Nagrada={agent_metrics['total_reward']:.2f}")
        
        # Sačuvaj rezultate
        results.append({
            'run': i,
            'seed': seed,
            'fixed_steps': fixed_metrics['total_steps'],
            'fixed_avg_waiting': fixed_metrics['avg_waiting'],
            'agent_steps': agent_metrics['total_steps'],
            'agent_avg_waiting': agent_metrics['avg_waiting'],
            'agent_reward': agent_metrics['total_reward'],
            'improvement_steps': fixed_metrics['total_steps'] - agent_metrics['total_steps'],
            'improvement_waiting': fixed_metrics['avg_waiting'] - agent_metrics['avg_waiting'],
            'fixed_avg_queue': fixed_metrics['avg_queue_length'],
            'agent_avg_queue': agent_metrics['avg_queue_length'],
        })
    
    # Sačuvaj i prikaži rezultate
    df = save_results(results)
    plot_results(df)
    plot_queue_lengths(df)
    
    # Prikaz ukupnog poboljšanja
    avg_step_improvement = df['improvement_steps'].mean()
    avg_waiting_improvement = df['improvement_waiting'].mean()
    
    print("\n" + "="*50)
    print(f"PROSEČNO POBOLJŠANJE U {NUM_EVAL_EPISODES} POKRETANJA:")
    print(f"Skraćenje trajanja simulacije: {avg_step_improvement:.1f} koraka ({avg_step_improvement/df['fixed_steps'].mean()*100:.1f}%)")
    print(f"Smanjenje vremena čekanja: {avg_waiting_improvement:.1f} sekundi ({avg_waiting_improvement/df['fixed_avg_waiting'].mean()*100:.1f}%)")
    print("="*50)