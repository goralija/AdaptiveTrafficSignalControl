# Identifikacija semafora
TL_ID = "cluster_1955770138_39630061_6970320248_6970320249_#1more"

# Vremenska ograničenja faza
MIN_PHASE_DURATION = 15
MAX_PHASE_DURATION = 110

# Konfiguracija simulacije
CONFIG_FILE = "../simulation-config/osm.sumocfg"
SIMULATION_FOLDER = "../simulation-config/"
NET_FILE = "osm.net.xml"
ROU_FILE = "routes.rou.xml"
SUMO_BINARY = "sumo"
SUMO_BINARY_EVAL = "sumo-gui"

# Parametri treniranja
MAX_STEPS = 9200  # Povećano na 2 sata simulacije
EPISODES_DONE = 0
NUM_EPISODES = 500
NUM_EVAL_EPISODES = 5
NUM_ROUTE_VARIATIONS = 2

# Hiperparametri Q-učenja
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.5
ALPHA_DECAY = 0.9999
EPSILON_DECAY = 0.995

# Putanje za čuvanje modela
Q_TABLE_PATH = "./q-tables-and-logs/qtable_final.pkl"

# Posljednje korištene vrijednosti (ažuriraju se automatski)
LAST_ALPHA = ALPHA
LAST_GAMMA = GAMMA
LAST_EPSILON = EPSILON

# Parametri generisanja ruta
SIM_START_OF_GENERATING = 0
SIM_GENERATING_RANGE_MIN = 1000
SIM_GENERATING_RANGE_MAX = 1500  
ROUTES_PER_SEC_RANGE_MIN = 2
ROUTES_PER_SEC_RANGE_MAX = 3
ROUTES_PER_SEC_RANGE_RANDOMIZE = False

# Konfiguracija nagrada (optimizovano)
REWARD_CONFIG = {
    'queue_weight': 0.4,
    'waiting_weight': 0.3,
    'flow_weight': 0.3,
    'queue_normalizer': 40,
    'waiting_normalizer': 80,
    'flow_normalizer': 10
}

last_alpha = 0.095123
last_gamma = 0.950000
last_epsilon = 0.040786
episodes_done = 500
