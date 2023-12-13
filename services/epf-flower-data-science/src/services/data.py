from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def download_iris_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='services/epf-flower-data-science/src/data', unzip=True)

    return {"status": "Dataset downloaded successfully"}

def load_iris_dataset():
    try:
        dataset = pd.read_csv('services/epf-flower-data-science/src/data/Iris.csv')
        return dataset
    except Exception as e:
        return {"error": f"Error loading dataset: {e}"}