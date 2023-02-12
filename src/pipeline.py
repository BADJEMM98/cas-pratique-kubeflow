from kfp.components import load_component_from_file
from kfp.compiler.compiler import Compiler 
import kfp.dsl as dsl



load_data_op = load_component_from_file("../components_yaml/load_data_component.yaml")
preprocess_data_op = load_component_from_file("../components_yaml/preprocess_data_component.yaml")
tunehp_xgboost_op = load_component_from_file("../components_yaml/tune_xgboost_component.yaml")
train_xgboost_op = load_component_from_file("../components_yaml/train_xgboost_component.yaml")
eval_xgboost_op = load_component_from_file("../components_yaml/eval_xgboost_component.yaml")
deploy_xgboost_op = load_component_from_file("../components_yaml/deploy_xgboost_component.yaml")


@dsl.pipeline(
    name="salary_prediction_pipeline",
    description="pipeline d'entrainement d'un modèle de prédiction de salaire",
)
def salary_prediction_pipeline(
    access_key_id:str,
    access_key_secret:str,
    filename:str,
    model_registry:str,

):
    load_data_task = load_data_op(access_key_id,access_key_secret,filename)
    preprocess_data_task = preprocess_data_op(load_data_task.outputs['data'])
    tunehp_xgboost_task = tunehp_xgboost_op(preprocess_data_task.outputs['train'],preprocess_data_task.outputs['test'])
    train_xgboost_task = train_xgboost_op(preprocess_data_task.outputs['train'],tunehp_xgboost_task.outputs["bestconfigs"],model_registry,access_key_id,access_key_secret)
    eval_xgboost_task = eval_xgboost_op(preprocess_data_task.outputs['test'],train_xgboost_task.outputs["modelname"],model_registry,access_key_id,access_key_secret)
    deploy_xgboost_task = deploy_xgboost_op()


Compiler.compile(pipeline_func=salary_prediction_pipeline, package_path="/pipeline.yaml")