# Smart Blood Bank Distribution

Local academic demo that uses reinforcement learning to coordinate synthetic blood procurement and distribution from one central bank to four hospitals. It does **not** use real hospital data.

## Run locally

Requires Python 3.11+ and Node 20+.

1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python -m backend.train` (CPU demo defaults to 30,000 steps)
5. `python -m backend.evaluate`
6. In terminal one: `uvicorn backend.main:app --reload --port 8000`
7. In terminal two: `cd frontend; npm install; npm run dev`

Open the Vite URL, normally http://localhost:5173. The API docs are at http://localhost:8000/docs.

## MDP formulation

- **State:** fresh/mid/near-expiry inventory for each blood group at the bank and hospitals, plus recent demand.
- **Action:** procurement at the bank plus allocations to each hospital.
- **Reward:** negative daily cost: expiry/wastage, shortage, inventory holding, and distance-weighted transport.

The environment uses FEFO issuing. The comparison policy is a documented order-up-to heuristic that replenishes low bank stock and sends hospitals towards a fixed target. Evaluation uses identical random seeds for both policies. Model outputs are generated in `backend/data/`; no benchmark claims are included here because results depend on the local training run.

## API

`GET /network`, `/training/history`, `/comparison`, `/inventory/timeseries`; `POST /simulate/run` runs a fresh fixed-seed evaluation.
