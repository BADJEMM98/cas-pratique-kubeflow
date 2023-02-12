from kfp.components import create_component_from_func,OutputPath,InputPath

def preprocess_data(data_path:InputPath('CSV'),train_path:OutputPath('CSV'),test_path:OutputPath('CSV')):
    import pandas as pd
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data = pd.read_csv(data_path)
    
    # Suppression des colonnes avec un fort déséquilibre et avec une valeur unique
    data.columns = data.columns.str.replace(' ', '')
    data.drop(columns=["race","native-country","capital-gain","capital-loss"],inplace=True)
    
    #Suppression des lignes dupliquées
    data.drop_duplicates(inplace=True)

    # CATEGORICAL/ONE-HOT ENCODING
    data["salary"] = data["salary"].apply(lambda x:x.strip())
    data["salary"] = data["salary"].replace(["<=50K", ">50K"],[0,1])

    data["sex"] = data["sex"].apply(lambda x:x.strip())
    data["sex"].replace(["Male","Female"],[0,1])

    data["workclass"].replace({"?":"Private"})
    data["workclass"] = data["workclass"].astype('category')
    data["workclass"] = data["workclass"].cat.codes

    data["education"] = data["education"].astype('category')
    data["education"] = data["education"].cat.codes

    data["occupation"].replace({"?":"Prof-specialty"})
    data["occupation"] = data["occupation"].astype('category')
    data["occupation"] = data["occupation"].cat.codes

    data["sex"] = data["sex"].replace(["Male","Female"],[0,1])

    data = pd.get_dummies(
        data=data,
        columns=["relationship","marital-status"],
        prefix=["rel","mar_s"]
        )
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv(train_path,index=False)
    test.to_csv(test_path,index=False)

if __name__ == "__main__":

    from kfp.components import create_component_from_func
    preprocess_data_op = create_component_from_func(
                    func=preprocess_data,
                    output_component_file="../components_yaml/preprocess_data_component.yaml",
                    base_image= "python:3.8",
                    packages_to_install=['pandas',"scikit-learn"]
    )