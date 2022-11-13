# MLOPS-Dagshub
trying out dagshub

Step 1: 
    Create Git repo 
    create DagsHub repo: https://dagshub.com

Step 2: 
    install DVC 
    dvc init 
configure dvc: 
    dvc remote add origin https://dagshub.com/sridevinarayan/MLOPS-Dagshub.dvc
    dvc remote modify origin --local auth basic 
    dvc remote modify origin --local user sridevinarayan 
    dvc remote modify origin --local password Sp@rky123

    dvc pull -r origin
    dvc add data/raw
    dvc push -r origin

install mlflow
#add the following in the python code!
mlflow.set_tracking_uri("https://dagshub.com/sridevinarayan/MLOPS-Dagshub.mlflow")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

export MLFLOW_TRACKING_USERNAME=sridevinarayan
export MLFLOW_TRACKING_PASSWORD=Sp@rky123
   