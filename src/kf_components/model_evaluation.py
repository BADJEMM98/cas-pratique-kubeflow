from kfp.components import InputPath
from typing import NamedTuple


def eval_xgboost(
    test_path:InputPath('CSV'),
    modelname:str,
    modelregistry:str,
    access_key_id:str,
    acces_key_secret:str,
)-> NamedTuple('EvalOutputs', [('eval_accuracy', float),('eval_roc_auc_score', float)]):
    
    import pandas as pd
    import boto3
    import tempfile
    import joblib
    from xgboost import XGBClassifier
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    test_data = pd.read_csv(test_path)
    
    X_test = test_data.drop(["salary"], axis=1)
    y_test = test_data["salary"].to_list()
    
    with tempfile.TemporaryFile() as fp:
        s3_client = boto3.client('s3',aws_access_key_id=access_key_id,aws_secret_access_key=acces_key_secret)
        s3_client.download_fileobj(Fileobj=fp, Bucket=modelregistry, Key=modelname)
        fp.seek(0)
        xgb_model = joblib.load(fp)
    
    y_pred = xgb_model.predict(X_test)
    roc_auc = roc_auc_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    
    output = namedtuple("EvalOutputs",["eval_accuracy","eval_roc_auc_score"])
    return output(accuracy, roc_auc)


if __name__ == "__main__":

    from kfp.components import create_component_from_func
    eval_xgboost_op = create_component_from_func(
        eval_xgboost, output_component_file="../components_yaml/eval_xgboost_component.yaml",    
        base_image= "python:3.8",
        packages_to_install = ["boto3","pandas","scikit-learn","xgboost"]
    )