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
    REWARD_CONFIG
)

check_sumo_home()

# Load learned Q-table
try:
    with open("q-tables-and-logs/qtable_final.pkl", "rb") as f:
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
    os.chdir(SIMULATION_FOLDER)
    sim_generating_end = generate_random_routes(seed)
    os.chdir("../src")
    
    # Pokreni SUMO
    traci.start([SUMO_BINARY_EVAL, "-c", CONFIG_FILE])
    
    step = 0
    last_action_time = 0
    cumulative_waiting = 0
    measurement_count = 0
    total_reward = 0
    total_departed = 0
    total_arrived = 0
    
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
        
        # Ako koristimo agenta, odredi akciju
        if use_agent:
            state = get_state(TL_ID)
            
            if step - last_action_time >= MIN_PHASE_DURATION:
                if step - last_action_time >= MAX_PHASE_DURATION:
                    action = 1
                else:
                    action = agent.choose_action(state)
                
                if action == 1:
                    current_phase = traci.trafficlight.getPhase(TL_ID)
                    new_phase = (current_phase + 1) % get_phase_count(TL_ID)
                    traci.trafficlight.setPhase(TL_ID, new_phase)
                    last_action_time = step
            
            # Izračunaj nagradu (samo za praćenje, ne za učenje)
            reward = calculate_reward(TL_ID, REWARD_CONFIG)
            total_reward += reward
    
    # Prikupi finalne metrike
    metrics['total_steps'] = step
    metrics['departed'] = departed
    metrics['arrived'] = arrived
    metrics['avg_waiting'] = cumulative_waiting / measurement_count if measurement_count > 0 else 0
    metrics['total_reward'] = total_reward
    
    traci.close()
    return metrics

def save_results(results, filename="evaluation_results.csv"):
    """Čuva rezultate evaluacije u CSV fajl"""
    os.makedirs("evaluation-results", exist_ok=True)
    path = os.path.join("evaluation-results", filename)
    
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
    plt.savefig("evaluation-results/evaluation_comparison.png")
    print("Grafikoni sačuvani kao evaluation_comparison.png")

if __name__ == "__main__":
    results = []
    
    for i in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n{'='*50}")
        print(f"Evaluacijsko pokretanje {i}/{NUM_EVAL_EPISODES}")
        
        # Generiši jedinstveni seed za obe varijante
        seed = random.randint(1, 10000) % NUM_ROUTE_VARIATIONS
        
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
            'improvement_waiting': fixed_metrics['avg_waiting'] - agent_metrics['avg_waiting']
        })
    
    # Sačuvaj i prikaži rezultate
    df = save_results(results)
    plot_results(df)
    
    # Prikaz ukupnog poboljšanja
    avg_step_improvement = df['improvement_steps'].mean()
    avg_waiting_improvement = df['improvement_waiting'].mean()
    
    print("\n" + "="*50)
    print(f"PROSEČNO POBOLJŠANJE U {NUM_EVAL_EPISODES} POKRETANJA:")
    print(f"Skraćenje trajanja simulacije: {avg_step_improvement:.1f} koraka ({avg_step_improvement/df['fixed_steps'].mean()*100:.1f}%)")
    print(f"Smanjenje vremena čekanja: {avg_waiting_improvement:.1f} sekundi ({avg_waiting_improvement/df['fixed_avg_waiting'].mean()*100:.1f}%)")
    print("="*50)