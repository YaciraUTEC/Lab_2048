print("Starting evaluation...")

import json
import os
from datetime import datetime
from solucion import Agent
from evaluation import evaluate_agent_scalar, evaluate_agent_scalar_render

def save_score(score, seeds_count, render_mode, filename="scores_history.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result_entry = {
        "timestamp": timestamp,
        "score": float(score),
        "seeds_count": seeds_count,
        "render_mode": render_mode
    }
    
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(result_entry)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Score guardado: {score} en {filename}")
    
    if len(history) > 1:
        best_score = max(entry["score"] for entry in history)
        print(f"Mejor score histórico: {best_score}")

EPISODES = 10 # Cambiar aquí para más/menos episodios
seeds = list(range(EPISODES))
print(f"Evaluando con {EPISODES} episodios...")

agent = Agent()

RENDER = True  # ponlo en True si quieres ventana (más lento con muchos episodios)
if RENDER:
	results = evaluate_agent_scalar_render(agent, seeds, step_delay=0.02)
else:
	results = evaluate_agent_scalar(agent, seeds)

print("Finished evaluation.")
print(f"Score actual: {results['final_score']}")
print(f"Detalles: {results}")

save_score(results['final_score'], len(seeds), RENDER)