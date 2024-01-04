from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import json
from src.schemas.message import MessageResponse
import joblib
from src.services.firestore import FirestoreClient






def download_iris_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='services/epf-flower-data-science/src/data', unzip=True)

    return {"status": "Dataset downloaded successfully"}

def load_iris_dataset():
    dataset = pd.read_csv('services/epf-flower-data-science/src/data/Iris.csv')
    return dataset.to_json(orient="records")

def preprocessing_data():
    dataset = pd.read_json(load_iris_dataset())
    dataset = dataset.drop("Id", axis=1)
    dataset['Species'] = dataset['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1,
                                                  'Iris-virginica': 2})
    return dataset.to_json(orient="records")

def train_test_split_data():
    """Split the data into train and test sets"""
    preprocessed_data = pd.read_json(preprocessing_data())
    data_train, data_test = train_test_split(preprocessed_data, test_size=0.2)
    return data_train.to_json(orient="records"), data_test.to_json(orient="records")

def train_model():
    """Train the model, saves it and saves the parameters of the model"""
    data_train = pd.read_json(train_test_split_data()[0])
    X_train = data_train.drop("Species", axis=1)
    y_train = data_train["Species"]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    params = model.get_params()
    params_path = os.path.join("services/epf-flower-data-science/src/config/", "model_parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f)

    os.makedirs("services/epf-flower-data-science/src/models/", exist_ok=True)
    model_path = os.path.join("services/epf-flower-data-science/src/models/model.joblib")
    joblib.dump(model, model_path)  

    return {"The model is trained"},model

def predict():
    """Make predictions"""
    model = train_model()[1]
    data_test = pd.read_json(train_test_split_data()[1])
    X_test = data_test.drop("Species", axis=1)
    y_pred = model.predict(X_test)
    return pd.DataFrame(y_pred).to_json(orient="records")

def retreive_firestore_parameters():
    """Retreive parameters from Firestone"""
    client = FirestoreClient()
    params = client.get(collection_name="parameters", document_id="parameters")
    return params

def update_firestore():
    """Update parameters on the Firestone database"""
    client = FirestoreClient()
    parameters_ref = client.client.collection("parameters").document("parameters")
    origin_params = client.get(collection_name="parameters", document_id="parameters")
    origin_params['n_estimators'] = 100
    origin_params['criterion'] = "gini"
    parameters_ref.set(origin_params)
    return origin_params