import os

from fastapi import FastAPI

from .predict import VespaTuneONNXPredict


app = FastAPI()
predictor = VespaTuneONNXPredict(model_path=os.environ.get("VESPATUNE_MODEL_PATH"))
schema = predictor.get_prediction_schema()


@app.post("/predict")
def predict(sample: schema):
    sample_dict = sample.model_dump()
    return predictor.predict_single(sample_dict)
