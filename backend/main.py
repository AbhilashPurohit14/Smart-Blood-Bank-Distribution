from __future__ import annotations
import json, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import BLOOD_GROUPS, DATA_DIR, DISTANCES_KM, NUM_HOSPITALS
from .env import BloodBankEnv
from .evaluate import evaluate

app = FastAPI(title='Smart Blood Bank Distribution API')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_methods=['*'], allow_headers=['*'])
def data(name):
    path = DATA_DIR / name
    if not path.exists(): raise HTTPException(404, 'No generated data yet. Run training and evaluation first.')
    return json.loads(path.read_text())
@app.get('/network')
def network():
    env = BloodBankEnv(); env.reset(seed=42)
    return {'nodes': [{'id':'bank','name':'Central Bank','kind':'bank','inventory':int(env.inventory[0].sum())}] +
      [{'id':f'h{i+1}','name':f'Hospital {i+1}','kind':'hospital','inventory':int(env.inventory[i+1].sum())} for i in range(NUM_HOSPITALS)],
      'distances_km': DISTANCES_KM, 'blood_groups': BLOOD_GROUPS}
@app.get('/training/history')
def training_history(): return data('training_history.json')
@app.get('/comparison')
def comparison(): return data('comparison.json')
@app.get('/inventory/timeseries')
def inventory(): return data('inventory_timeseries.json')
@app.post('/simulate/run')
def simulate_run():
    try: result = evaluate()
    except FileNotFoundError as exc: raise HTTPException(409, str(exc))
    return {'id': str(uuid.uuid4()), 'comparison': result}
