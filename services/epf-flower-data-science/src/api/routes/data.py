from fastapi import APIRouter, HTTPException
from src.services.data import download_iris_dataset, load_iris_dataset


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