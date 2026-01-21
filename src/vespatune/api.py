import os
import shutil
from typing import List
import json
import asyncio
import pandas as pd
import optuna

from fastapi import (
    FastAPI,
    Request,
    BackgroundTasks,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .predict import VespaTuneONNXPredict
from .core import VespaTune
from .models import list_models
from .enums import TaskType


app = FastAPI()


# Global state
ACTIVE_STUDY_PATH = None
STOP_TRAINING = False
IS_TRAINING = False

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Templates
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

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
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- Optuna Callback ---
def optuna_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        msg = {
            "type": "trial_complete",
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        # We need a running event loop to send async messages from a sync callback
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.create_task(manager.broadcast(msg))


class TrainRequest(BaseModel):
    train_filename: str
    valid_filename: str
    output_dir: str
    target_columns: str
    id_column: str = "id"
    task: str = "classification"
    model_type: str = "xgboost"
    num_trials: int = 100
    time_limit: int = 3600


def run_training(config: TrainRequest):
    # Notify start
    asyncio.run(
        manager.broadcast({"type": "status", "message": "Initializing training..."})
    )

    targets = [x.strip() for x in config.target_columns.split(";")]
    try:
        vt = VespaTune(
            train_filename=config.train_filename,
            valid_filename=config.valid_filename,
            output=config.output_dir,
            task=config.task,
            idx=config.id_column,
            targets=targets,
            model_type=config.model_type,
            num_trials=config.num_trials,
            time_limit=config.time_limit,
        )
        # Notify processing data
        asyncio.run(
            manager.broadcast({"type": "status", "message": "Processing data..."})
        )

        # Reset stop signal
        global STOP_TRAINING, ACTIVE_STUDY_PATH, IS_TRAINING
        STOP_TRAINING = False
        IS_TRAINING = True
        ACTIVE_STUDY_PATH = f"{config.output_dir}/params.db"

        # Callbacks
        def optuna_check_stop(study, trial):
            if STOP_TRAINING:
                study.stop()
                raise optuna.exceptions.TrialPruned("Training cancelled by user")

        def simple_check_stop():
            if STOP_TRAINING:
                raise optuna.exceptions.TrialPruned("Training cancelled by user")

        vt.train(callbacks=[optuna_callback, optuna_check_stop], check_stop=simple_check_stop)

        # Notify complete
        asyncio.run(
            manager.broadcast(
                {"type": "status", "message": "Training completed successfully!"}
            )
        )

    except optuna.exceptions.TrialPruned as e:
         asyncio.run(
            manager.broadcast({"type": "status", "message": "Training cancelled"})
        )
         asyncio.run(manager.broadcast({"type": "info", "message": str(e)}))

    except Exception as e:
        asyncio.run(manager.broadcast({"type": "error", "message": str(e)}))

    finally:
        IS_TRAINING = False
        asyncio.run(manager.broadcast({"type": "training_finished"}))


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
    file_location = os.path.join(UPLOAD_DIR, file.filename)
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
def get_current_study():
    return {"db_path": ACTIVE_STUDY_PATH, "is_training": IS_TRAINING}


@app.post("/stop")
def stop_training():
    global STOP_TRAINING
    STOP_TRAINING = True
    return {"message": "Stopping training..."}


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
                        "datetime_start": (
                            t.datetime_start.isoformat() if t.datetime_start else None
                        ),
                        "datetime_complete": (
                            t.datetime_complete.isoformat()
                            if t.datetime_complete
                            else None
                        ),
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


@app.post("/train")
async def train(config: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training, config)
    return {"message": "Training started", "config": config}


@app.post("/predict")
def predict(sample: dict):
    if not predictor:
        return {"error": "Model not loaded"}
    return predictor.predict_single(sample)
