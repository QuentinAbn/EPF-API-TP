from fastapi import APIRouter, HTTPException
from src.services.data import *

router = APIRouter()

@router.get("/data")
def download_iris():
    return download_iris_dataset()

@router.get("/load_data")
def get_iris_dataset():
    dataset = load_iris_dataset()
    if "error" in dataset:
        raise HTTPException(status_code=404, detail=dataset["error"])
    return dataset

@router.get("/preprocessed_data")
def preprocessed_dataset():
    return preprocessing_data()

@router.get("/split_data")
def split_dataset():
    return train_test_split_data()

@router.get("/model_train")
def model_training():
    return train_model()[0]

@router.get("/predictions")
def model_predictions():
    return predict()