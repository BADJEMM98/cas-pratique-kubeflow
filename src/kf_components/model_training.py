from kfp.components import create_component_from_func,InputPath
from typing import NamedTuple

def train_xgboost(
    train_path:InputPath('CSV'),
    bestconfigs_path:InputPath('json'),
    bucket:str,
    access_key_id:str,
    acces_key_secret:str,
)-> NamedTuple('TrainOutputs', [('train_accuracy', float),('train_roc_auc_score', float),('modelname', str)]):
    
    from datetime import datetime
    import pandas as pd
    import boto3
    import json
    from xgboost import XGBClassifier
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, roc_auc_score

    model_name = "xgboost"

    time = str(datetime.now()).replace(" ", "_")
    model_path = f"{model_name}_{time}.sbt"

    modelname = f"/models/{model_path}"
    
    train_data = pd.read_csv(train_path)
    
    X_train = train_data.drop(["salary"], axis=1)
    y_train = train_data["salary"]
    
    config = json.load(bestconfigs_path)

    xgb_model = XGBClassifier(
        random_state=42, 
        n_estimators =config['n_estimators'], 
        max_depth = int(config['max_depth']), 
        gamma = config['gamma'],
        booster = config['booster'],
        learning_rate = config["lr"]
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_train)
    roc_auc = roc_auc_score(y_train,y_pred)
    accuracy = accuracy_score(y_train,y_pred)
    
    xgb_model.save_model(model_path)
    s3_client = boto3.client('s3',aws_access_key_id=access_key_id,aws_secret_access_key=acces_key_secret)
    response= s3_client.upload_file(model_path, bucket, f"models/{model_path}")
    
    output = namedtuple("TrainOutputs",["train_accuracy","train_roc_auc_score","modelname"])
    
    return output(accuracy, roc_auc,modelname)

if __name__ == "__main__":

    from kfp.components import create_component_from_func
    
    train_xgboost_op = create_component_from_func(
        train_xgboost, output_component_file="../components_yaml/train_xgboost_component.yaml",    
        base_image= "python:3.8",
        packages_to_install = ["boto3","pandas","scikit-learn","xgboost"]
    )