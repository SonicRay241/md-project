from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
import pandas as pd

from model.classifier import Model
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    file_path = "./model/model_bundle.pkl"

    # Unpack
    models["obesity_predictor"] = Model.load(file_path)

    yield
    # Clean up the ML models and release the resources
    models.clear()

# Init
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return "Use /predict"

class ModelInput(BaseModel):
    gender: Literal["Male"] | Literal["Female"]
    age: int
    height: float
    weight: float
    family_history_with_overweight: Literal["yes"] | Literal["no"]
    favc: Literal["yes"] | Literal["no"]
    fcvc: float
    ncp: float
    caec: Literal["no"] | Literal["Sometimes"] | Literal["Frequently"] | Literal["Always"]
    smoke: Literal["yes"] | Literal["no"]
    ch20: float
    scc: Literal["yes"] | Literal["no"]
    faf: float
    tue: float
    calc: Literal["no"] | Literal["Sometimes"] | Literal["Frequently"] | Literal["Always"]
    mtrans: Literal["Public_Transportation"] | Literal["Automobile"] | Literal["Bike"] | Literal["Walking"] | Literal["Motorbike"]

@app.post("/predict")
def predict(params: ModelInput):
    input_df = pd.DataFrame(
        columns=["Gender","Age","Height","Weight","family_history_with_overweight","FAVC","FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"],
        data=[list(params.model_dump().values())]
    )

    return {
        "prediction": models["obesity_predictor"].predict(input_df)[0]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)