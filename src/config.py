# Identifikacija semafora
TL_ID = "cluster_1955770138_39630061_6970320248_6970320249_#1more"

# Vremenska ograničenja faza
MIN_PHASE_DURATION = 5
MAX_PHASE_DURATION = 110

# Konfiguracija simulacije
CONFIG_FILE = "../simulation-config/osm.sumocfg"
SIMULATION_FOLDER = "../simulation-config/"
NET_FILE = "osm.net.xml"
ROU_FILE = "routes.rou.xml"
SUMO_BINARY = "sumo"
SUMO_BINARY_EVAL = "sumo"

# Parametri treniranja
MAX_STEPS = 22222  
EPISODES_DONE = 1500
NUM_EPISODES = 2500
NUM_EVAL_EPISODES = 50
NUM_ROUTE_VARIATIONS = 7

# Hiperparametri Q-učenja
ALPHA = 0.187
GAMMA = 0.95
EPSILON = 1.0
ALPHA_DECAY = 0.9996
EPSILON_DECAY = 0.997

# Putanje za čuvanje modela
Q_TABLE_PATH = "./q-tables-and-logs/qtable_final.pkl"
EVAL_Q_TABLE_PATH = "q-tables-and-logs/tables/qtable_ep"

# Parametri generisanja ruta
SIM_START_OF_GENERATING = 0
SIM_GENERATING_RANGE_MIN = 800
SIM_GENERATING_RANGE_MAX = 1100  
ROUTES_PER_SEC_RANGE_MIN = 0.7
ROUTES_PER_SEC_RANGE_MAX = 1.1
ROUTES_PER_SEC_RANGE_RANDOMIZE = False

last_alpha = 0.060156
last_gamma = 0.950000
last_epsilon = 0.001538
episodes_done = 2199
