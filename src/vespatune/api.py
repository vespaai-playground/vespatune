import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import List

import optuna
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .db import create_run, delete_run, get_active_run, get_all_runs, get_run, init_db, update_run_status
from .enums import TaskType
from .models import list_models
from .predict import VespaTuneONNXPredict


# Helper: Check if process is alive
def is_process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 checks if process exists
        return True
    except (ProcessLookupError, OSError):
        return False


# Helper: Check if final model exists for a run
def has_final_model(output_dir):
    if not output_dir or not os.path.exists(output_dir):
        return False
    final_model_path = os.path.join(output_dir, "vtune_model.final")
    return os.path.exists(final_model_path)


# Startup: Recover stale runs
def recover_stale_runs():
    runs = get_all_runs()
    for run in runs:
        if run["status"] in ["running", "pending"]:
            pid = run.get("pid")
            run_id = run["id"]

            # Check if process is still alive
            if is_process_alive(pid):
                continue  # Still running, leave it

            # Process is dead - check if it completed
            if has_final_model(run["output_dir"]):
                update_run_status(run_id, "completed")
                print(f"Run {run_id}: Marked as completed (final model exists)")
            else:
                update_run_status(run_id, "error", error="Process terminated unexpectedly")
                print(f"Run {run_id}: Marked as error (no final model)")


# Initialize DB on startup
init_db()
recover_stale_runs()

app = FastAPI()


# Graceful shutdown: Stop all active runs
def cleanup_active_runs():
    print("Shutting down: Stopping all active runs...")
    runs = get_all_runs()
    for run in runs:
        if run["status"] in ["running", "pending"] and run.get("pid"):
            try:
                os.kill(run["pid"], signal.SIGTERM)
                update_run_status(run["id"], "stopped")
                print(f"Stopped run {run['id']}")
            except (ProcessLookupError, OSError):
                print(f"Run {run['id']} process already dead")


# Register signal handlers
def signal_handler(signum, frame):
    cleanup_active_runs()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

model_path = os.environ.get("VESPATUNE_MODEL_PATH")
if model_path and os.path.exists(model_path):
    predictor = VespaTuneONNXPredict(model_path=model_path)
    schema = predictor.get_prediction_schema()
else:
    predictor = None
    schema = None


# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Start a background task to stream logs and monitor trials if a run is active
    active_run = get_active_run()
    log_task = None
    trial_task = None
    if active_run and active_run["status"] == "running":
        log_task = asyncio.create_task(stream_logs(active_run["output_dir"], manager))
        trial_task = asyncio.create_task(monitor_trials(active_run["db_path"], manager))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if log_task:
            log_task.cancel()
        if trial_task:
            trial_task.cancel()


async def stream_logs(output_dir: str, manager: ConnectionManager):
    log_file = os.path.join(output_dir, "run.log")

    # Wait for file to exist
    while not os.path.exists(log_file):
        await asyncio.sleep(1)
        # Check if run is still alive? (For simplified logic, just wait a bit)

    # Tail the file
    with open(log_file, "r") as f:
        # Go to end
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue

            # Simple parsing or just send raw line
            await manager.broadcast({"type": "info", "message": line.strip()})


async def monitor_trials(db_path: str, manager: ConnectionManager):
    storage = f"sqlite:///{db_path}"
    last_trial_count = 0
    study_name = "vespatune"  # Assuming default name from core.py

    while True:
        try:
            # Check if DB exists yet
            if not os.path.exists(db_path):
                await asyncio.sleep(2)
                continue

            # Load study purely to check trials
            # Note: optimistically loading study.
            # In production, we might want a lighter query or share the same storage obj
            study = optuna.load_study(study_name=study_name, storage=storage)
            trials = study.trials

            # Filter for completed trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            current_count = len(completed_trials)

            if current_count > last_trial_count:
                # Broadcast new trials
                # We sort by number to ensure order
                completed_trials.sort(key=lambda t: t.number)

                # Send only the new ones
                for i in range(last_trial_count, current_count):
                    t = completed_trials[i]
                    msg = {
                        "type": "trial_complete",
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "user_attrs": t.user_attrs,
                        "best_value": study.best_value,
                        "best_params": study.best_params,
                    }
                    await manager.broadcast(msg)

                last_trial_count = current_count

            await asyncio.sleep(2)

        except Exception:
            # Logic to handle race conditions or DB locks
            await asyncio.sleep(2)


class TrainRequest(BaseModel):
    project_name: str
    train_filename: str
    valid_filename: str
    target_columns: str
    id_column: str = "id"
    task: str = "classification"
    model_type: str = "xgboost"
    num_trials: int = 100
    time_limit: int = 3600


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": list_models(),
            "tasks": TaskType.list_str(),
        },
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = os.path.expanduser("~/.vespatune/uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Add unique prefix to avoid filename collisions
    unique_id = f"{int(time.time() * 1000)}"
    safe_filename = f"{unique_id}_{file.filename}"
    file_location = os.path.join(upload_dir, safe_filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Analyze columns
    df = pd.read_csv(file_location, nrows=5)
    columns = list(df.columns)

    return {"filename": file.filename, "path": file_location, "columns": columns}


@app.get("/columns")
def get_columns(path: str):
    if not os.path.exists(path):
        return {"error": "File not found"}

    try:
        df = pd.read_csv(path, nrows=5)
        return {"columns": list(df.columns)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/current_study")
def get_current_study_endpoint():
    run = get_active_run()
    if run:
        config = json.loads(run["config"]) if run["config"] else {}
        return {
            "db_path": run["db_path"],
            "is_training": run["status"] == "running",
            "run_id": run["id"],
            "config": config,
        }
    return {"db_path": None, "is_training": False, "config": None}


@app.post("/stop")
def stop_training():
    run = get_active_run()
    if not run or run["status"] != "running":
        return {"message": "No active training run"}

    pid = run["pid"]
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            update_run_status(run["id"], "stopped")
            return {"message": "Stop signal sent to process"}
        except ProcessLookupError:
            update_run_status(run["id"], "error", error="Process not found")
            return {"message": "Process not found, status updated"}
        except Exception as e:
            return {"error": str(e)}

    return {"message": "No PID associated with run"}


@app.get("/study/{study_name}")
def get_study(study_name: str, db_path: str):
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trials.append(
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "datetime_start": (t.datetime_start.isoformat() if t.datetime_start else None),
                        "datetime_complete": (t.datetime_complete.isoformat() if t.datetime_complete else None),
                        "duration": t.duration.total_seconds() if t.duration else None,
                        "user_attrs": t.user_attrs,
                    }
                )
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": trials,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/meta")
def get_meta():
    if schema:
        return schema.model_json_schema()
    return {}


@app.get("/runs")
def list_runs():
    runs = get_all_runs()
    results = []
    for r in runs:
        conf = json.loads(r["config"]) if r["config"] else {}
        results.append(
            {
                "id": r["id"],
                "status": r["status"],
                "created_at": r["created_at"],
                "model_type": conf.get("model_type", "unknown"),
                "task": conf.get("task", "unknown"),
                "output_dir": r["output_dir"],
            }
        )
    return {"runs": results}


@app.get("/runs/{run_id}")
def get_run_details(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    config = json.loads(run["config"]) if run["config"] else {}
    return {
        "id": run["id"],
        "status": run["status"],
        "created_at": run["created_at"],
        "output_dir": run["output_dir"],
        "db_path": run["db_path"],
        "config": config,
        "pid": run["pid"],
    }


@app.get("/runs/{run_id}/trials")
def get_run_trials(run_id: int):
    run = get_run(run_id)
    if not run or not run["db_path"]:
        return {"trials": []}

    db_path = run["db_path"]
    if not os.path.exists(db_path):
        return {"trials": []}

    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name="vespatune", storage=storage)
        trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trials.append(
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "user_attrs": t.user_attrs,
                        "state": t.state.name,
                    }
                )
        # Sort by number
        trials.sort(key=lambda x: x["number"])

        return {
            "trials": trials,
            "best_value": study.best_value if len(trials) > 0 else None,
            "best_params": study.best_params if len(trials) > 0 else None,
        }
    except Exception as e:
        print(f"Error loading study for run {run_id}: {e}")
        return {"trials": []}


@app.get("/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: int):
    """List all artifacts for a run."""
    run = get_run(run_id)
    if not run or not run["output_dir"]:
        return {"artifacts": []}

    output_dir = run["output_dir"]
    if not os.path.exists(output_dir):
        return {"artifacts": []}

    artifacts = []

    # Main artifacts
    artifact_files = [
        ("vtune_model.final", "Final Model", "model"),
        ("vtune.config", "Config", "config"),
        ("vtune.best_params", "Best Parameters", "params"),
        ("vtune.categorical_encoder", "Categorical Encoder", "encoder"),
        ("vtune.target_encoder", "Target Encoder", "encoder"),
    ]

    for filename, label, category in artifact_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            artifacts.append(
                {
                    "name": filename,
                    "label": label,
                    "category": category,
                    "size": size,
                    "path": filepath,
                }
            )

    # ONNX artifacts
    onnx_dir = os.path.join(output_dir, "onnx")
    if os.path.exists(onnx_dir):
        for filename in os.listdir(onnx_dir):
            filepath = os.path.join(onnx_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                label = "ONNX Model" if filename.endswith(".onnx") else filename
                artifacts.append(
                    {
                        "name": f"onnx/{filename}",
                        "label": label,
                        "category": "onnx",
                        "size": size,
                        "path": filepath,
                    }
                )

    return {"artifacts": artifacts}


@app.get("/runs/{run_id}/artifacts/download")
def download_artifact(run_id: int, path: str):
    """Download a specific artifact."""
    run = get_run(run_id)
    if not run or not run["output_dir"]:
        raise HTTPException(status_code=404, detail="Run not found")

    # Security: ensure path is within output_dir
    output_dir = os.path.realpath(run["output_dir"])
    filepath = os.path.realpath(path)

    if not filepath.startswith(output_dir):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    filename = os.path.basename(filepath)
    return FileResponse(filepath, filename=filename)


@app.get("/active_run_id")
def get_active_run_id_endpoint():
    run = get_active_run()
    if run and run["status"] in ["running", "pending"]:
        return {"active_run_id": run["id"]}
    return {"active_run_id": None}


@app.delete("/runs/{run_id}")
def delete_run_endpoint(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Prevent deleting active runs
    if run["status"] in ["running", "pending"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running or pending job. Stop it first.",
        )

    # Cleanup Files
    try:
        if run["output_dir"] and os.path.exists(run["output_dir"]):
            shutil.rmtree(run["output_dir"])

        # db_path might be inside output_dir, but if it's separate check it
        if run["db_path"] and os.path.exists(run["db_path"]):
            # Verify it's a file before removing, just in case
            if os.path.isfile(run["db_path"]):
                os.remove(run["db_path"])
    except Exception as e:
        print(f"Error cleaning up files for run {run_id}: {e}")
        # We continue to delete from DB even if file cleanup fails partially

    delete_run(run_id)
    return {"status": "deleted"}


@app.post("/train")
async def train(config: TrainRequest):
    # Validate project name
    if not re.match(r"^[a-zA-Z0-9-]+$", config.project_name):
        return JSONResponse(
            status_code=400,
            content={"detail": "Project name must be alphanumeric with hyphens only"},
        )

    # Check if existing run is active
    active = get_active_run()
    if active and active["status"] == "running":
        return JSONResponse(status_code=400, content={"detail": "Training already in progress"})

    # Create output directory in ~/.vespatune/
    base_dir = os.path.expanduser("~/.vespatune")
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, config.project_name)

    if os.path.exists(output_dir):
        # Append timestamp to make unique
        timestamp = int(time.time())
        output_dir = os.path.join(base_dir, f"{config.project_name}-{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    db_path = f"{output_dir}/params.db"

    # Create DB entry
    run_id = create_run(config.dict(), config.project_name, output_dir, db_path)

    # Prepare config for worker (add generated output_dir)
    worker_config = config.dict()
    worker_config["output_dir"] = output_dir

    # Spawn worker process
    # We use python -m vespatune.worker
    cmd = [
        sys.executable,
        "-m",
        "vespatune.worker",
        "--run-id",
        str(run_id),
        "--config",
        json.dumps(worker_config),
    ]

    try:
        process = subprocess.Popen(cmd, start_new_session=True)
        # Update DB with PID
        update_run_status(run_id, "running", pid=process.pid)

        return {"message": "Training started", "run_id": run_id, "pid": process.pid}
    except Exception as e:
        update_run_status(run_id, "error", error=str(e))
        return JSONResponse(status_code=500, content={"detail": f"Failed to spawn worker: {e}"})


@app.post("/predict")
def predict(sample: dict):
    if not predictor:
        return {"error": "Model not loaded"}
    return predictor.predict_single(sample)
