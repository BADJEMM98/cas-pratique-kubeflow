from kfp.components import create_component_from_func,OutputPath


def load_data(access_key_id:str,acces_key_secret:str,filename:str,data_path:OutputPath('CSV')):
    import boto3
    import pandas as pd
    
    bucket_name = "fyc-bucket"
    s3_client = boto3.client('s3',aws_access_key_id=access_key_id,aws_secret_access_key=acces_key_secret)
    response = s3_client.get_object(Bucket=bucket_name, Key=filename)
    
    status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status_code == 200:
        print("Données chargées avec succès")
        data = pd.read_csv(response.get("Body"))
        data.to_csv(data_path,index=False)
        print(data)
    else:
        print("Echec du chargement des données")

if __name__ == "__main__":

    from kfp.components import create_component_from_func

    load_data_op = create_component_from_func(
                    func=load_data,
                    output_component_file="../components_yaml/load_data_component.yaml",
                    base_image= "python:3.8",
                    packages_to_install=['boto3','pandas']
    )