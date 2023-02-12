import textwrap


sa_yaml = textwrap.dedent("""\
apiVersion: "v1"
kind: ServiceAccount
metadata:
  name: my-sa
  namespace: kubeflow-user-example-com
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::136776963499:role/eks_s3_role" # replace with your IAM role ARN
    serving.kserve.io/s3-endpoint: "s3.amazonaws.com" # replace with your s3 endpoint e.g minio-service.kubeflow:9000
    serving.kserve.io/s3-usehttps: "1" # by default 1, if testing with minio you can set to 0
    serving.kserve.io/s3-region: "eu-west-3"
    serving.kserve.io/s3-useanoncredential: "false" # omitting this is the same as false, if true will ignore provided credential and use anonymous credentials
""")
f = open("sa.yaml", "w")
f.write(sa_yaml)
f.close()


xgboost_yaml = textwrap.dedent("""\
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "xgboostsrv"
  namespace: kubeflow-user-example-com
spec:
  predictor:
      serviceAccountName: my-sa
      xgboost:  
          storageUri: "s3://fycmodelsprod/new"
""")
f = open("xgboost_model.yaml", "w")
f.write(xgboost_yaml)
f.close()