from kfp.components import create_component_from_func,OutputPath,InputPath
from typing import NamedTuple

def deploy_xgboost(
    modelname:str,
    access_key_id:str,
    acces_key_secret:str
)-> NamedTuple('DeployOutputs', [('modelurl', str)]):
    import boto3
    import os
    import requests
    from collections import namedtuple
    
    s3_client = boto3.client('s3',aws_access_key_id=access_key_id,aws_secret_access_key=acces_key_secret)
    object_name = os.path.basename(modelname)
    source_url = f"https://s3.amazonaws.com/{modelname}"
    destination_bucket = 'fycmodelsprod'
    destination_object = object_name

    response = requests.get(source_url)

    # upload the object to the destination bucket
    response = s3_client.put_object(Bucket=destination_bucket, Key=destination_object, Body=response.content)

    new_url = ""
    print(f"Object from URL {source_url} has been successfully uploaded to {destination_object} in bucket {destination_bucket}.")
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Object {destination_object} has been successfully uploaded to bucket {destination_bucket}.")
        new_url = f"https://s3.amazonaws.com/{destination_bucket}/{object_name}"
    else:
        print(f"Failed to upload object {destination_object} to bucket {destination_bucket}. Error code: {response['ResponseMetadata']['HTTPStatusCode']}.")
    output = namedtuple("UploadOutputs",["modelurl"])

    return output(new_url)

if __name__ == "__main__":

    from kfp.components import create_component_from_func
    deploy_xgboost_op = create_component_from_func(
        deploy_xgboost, output_component_file="../components_yaml/deploy_xgboost_component.yaml",    
        base_image= "python:3.8",
        packages_to_install = ["boto3","requests"]
    )