from kfp.components import create_component_from_func,OutputPath,InputPath
from typing import NamedTuple, Dict

def tunehp_xgboost(
    train_path:InputPath('CSV'),
    test_path:InputPath('CSV'),
)-> NamedTuple('TuneOutputs',[("n_estimators", int),("learning_rate", float),('max_depth', int),('booster',str)]):
    
    from ray import tune
    import json
    import pandas as pd
    from xgboost import XGBClassifier
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop(["salary"], axis=1)
    y_train = train_data["salary"]
    
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(["salary"], axis=1)
    y_test = test_data["salary"]
    
    
    def objective(config):
        
        xgb_model = XGBClassifier(
            random_state=42, 
            n_estimators =config['n_estimators'], 
            max_depth = int(config['max_depth']), 
            booster = config['booster'],
            learning_rate = config["lr"]
            
        )
        
        xgb_model.fit(X_train, y_train,
                eval_set=[(X_test,y_test)], eval_metric="auc",
                early_stopping_rounds=5,verbose=False)
        
        y_pred = xgb_model.predict(X_test)
        roc_auc = roc_auc_score(y_test,y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        
        return {"roc_auc":roc_auc,"accuracy":accuracy}
        
        
    search_space = {
        "lr": tune.grid_search([0.1,0.02,0.005,0.001]),
        "n_estimators": tune.grid_search([100,300,200]),
        "booster": tune.grid_search(['gbtree', 'gblinear',"dart"]),
        "max_depth" : tune.grid_search([3,5,7,10])
    }

    tuner = tune.Tuner(
        objective,
        param_space=search_space,
    )
    
    results = tuner.fit()
    print(results.get_best_result(metric="roc_auc",mode="max").config)
    results_config = results.get_best_result(metric="roc_auc",mode="max").config
    
    n_estimators = results_config['n_estimators']
    max_depth = results_config['max_depth']
    booster = results_config['booster']
    learning_rate = results_config["lr"]
    
    output = namedtuple('TuneOutputs', ["n_estimators","learning_rate",'max_depth','booster'])
    print("n_estimators :",n_estimators)
    print("max_depth :",max_depth)
    print("booster :",booster)
    print("learning_rate :",learning_rate)
    
    return output(n_estimators,learning_rate,max_depth,booster)
    
if __name__ == "__main__":

    from kfp.components import create_component_from_func

    tunehp_xgboost_op = create_component_from_func(
        tunehp_xgboost, output_component_file="../components_yaml/tune_xgboost_component.yaml",    
        base_image= "python:3.8",
        packages_to_install = ["ray[tune]","pandas","scikit-learn","xgboost"]
    )