# Adaptivno upravljanje saobraćajnom signalizacijom pomoću Q-learninga

Ovaj repozitorij sadrži programski kod i konfiguracione fajlove korištene u okviru završnog rada na Elektrotehničkom fakultetu u Sarajevu.  
Cilj projekta je implementacija **Q-learning agenta** koji adaptivno upravlja semaforima na raskrsnici, u svrhu smanjenja gužvi i poboljšanja protočnosti saobraćaja.

---

## 🎯 Cilj i motivacija
Tradicionalni sistemi semafora koriste unaprijed definisane fiksne planove, što u realnim uslovima dovodi do zagušenja i povećanog vremena čekanja.  
Korištenjem **reinforcement learninga (pojačanog učenja)** moguće je razviti agenta koji se prilagođava trenutnom stanju u saobraćaju i donosi bolje odluke u realnom vremenu.

---

## 🧠 Metodologija
- **Algoritam:** Q-learning (model-free RL).  
- **Okruženje:** SUMO (Simulation of Urban Mobility) simulator + TraCI API za interakciju u realnom vremenu.  
- **Predstavljanje stanja:**  
  - Broj vozila po traci (diskretizovan u binove od 5 vozila, do max. 60).  
  - Trajanje trenutne faze semafora (diskretizovano u intervale od 10 sekundi).  
- **Akcije:**  
  1. Zadrži trenutnu fazu semafora  
  2. Promijeni fazu semafora  
- **Funkcija nagrade:** negativna vrijednost kvadratne sume dužina redova čekanja + penalizacija za neefikasne faze.  
- **Proces treniranja:** preko 2000 epizoda simulacije, sa periodičnim čuvanjem Q-tabela i logova.  
- **Evaluacija:** poređenje sa fiksnim semaforskim planom pomoću metrika prosječnog čekanja i ukupnog vremena trajanja simulacije.

---

## 📊 Rezultati
- RL agent značajno smanjuje prosječno vrijeme čekanja vozila u poređenju sa fiksnim planovima.  
- Eksperimenti pokazuju poboljšanje protočnosti i smanjenje zagušenja, čak i uz jednostavnu implementaciju Q-learninga.  
- Logovi i Q-tabele omogućavaju praćenje procesa učenja i analizu konvergencije.

---

## 📂 Struktura repozitorija

```
.
├── README.md # Ovaj fajl
├── requirements.txt # Python zavisnosti
│
├── simulation-config/ # SUMO konfiguracija
│ ├── osm.net.xml.gz # Saobraćajna mreža (kompresovana)
│ ├── osm.netccfg # Network konfiguracija
│ ├── osm.poly.xml.gz # Poligoni (objekti okruženja)
│ ├── osm.polycfg # Konfiguracija poligona
│ ├── osm.sumocfg # Glavni konfiguracioni fajl simulacije
│ ├── osm.view.xml # Podešavanja vizualizacije
│ └── osm_bbox.osm.xml.gz # OSM ulazni fajl za simulaciju
│
└── src/ # Python kod
├── config.py # Konfiguracija hiperparametara i simulacije
├── evaluate_agent.py # Evaluacija naučenog modela
├── run_training.py # Glavna skripta za trening
└── utils.py # Pomoćne funkcije
```

---

## ⚙️ Instalacija i pokretanje

### 1. Instalacija zavisnosti
Potrebno je imati **Python 3.9+** i instaliran **SUMO** simulator.  
Instalacija Python zavisnosti:
```bash
pip install -r requirements.txt
```

Instalacija SUMO-a:  
👉 [Uputstvo](https://sumo.dlr.de/docs/Installing.html)

### 2. Pokretanje treninga agenta
```bash
python src/run_training.py
```

### 3. Evaluacija naučenog modela
```bash
python src/evaluate_agent.py
```

---

## 🚀 Budući rad
- Proširenje na **više raskrsnica** (multi-agent RL).  
- Optimizacija **funkcije nagrade** (uključivanje protoka i kašnjenja).  
- Ispitivanje naprednijih RL algoritama: **DQN, DDPG, PPO**.  
- Automatska optimizacija hiperparametara.  

---

## 📚 Reference
- Abdulhai, B., Pringle, R., Karakoulas, G. J. (2003). *Reinforcement learning for true adaptive traffic signal control*.  
- Yau, K.-L. A., et al. (2017). *A survey on reinforcement learning models and algorithms for traffic signal control*.  
- Balaji, P., German, X., Srinivasan, D. (2010). *Urban traffic signal control using reinforcement learning agents*.  

---

✍️ Autor: **Harun Goralija**  
📅 2025
