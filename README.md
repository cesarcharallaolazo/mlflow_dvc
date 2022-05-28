# MLFLOW & DVC
Mlflow and Dvc and with docker

## Steps of this demo:

### Add a file to track & Push to DVC storage
- install dvc: brew install dvc
- dvc remote add -d dvc-ccho /Users/user/Documents/docs_cesar/other_repos/mlflow_dvc/external_storage
- show remote storages: cat .dvc/config 
- create train-data in : train_data/wine-quality.csv
- start tracking train-data with dvc: dvc add train_data/wine-quality.csv
- in train_data folder: train_data/wine-quality.csv.dvc was created. Also, in train_data folder: a .gitignore file was created to not track the wine-quality.csv to git
- add to git associated .dvc file and the created .gitignore & commit them
- create a tag for later tracking of the first version of your data: git tag -a 'v1' -m 'version-1 wine-quality.csv'
- push the first version of your data to dvc storage: dvc push
- verify push to dvc: ls -lR external_storage/

### Test dvc functionality by removing the dataset
- remove the train-data file: rm -rf train_data/wine-quality.csv
- also, remove from the cache: rm -rf .dvc/cache
- bring back the file: dvc pull
- verify train_data/wine-quality.csv is back

### New version of the dataset
- alter the file by removing some rows: sed -i '.old' '2005,3004d' train_data/wine-quality.csv
- add to dvc (repeated procedure): dvc add train_data/wine-quality.csv
- add to git associated .dvc file (repeated procedure): git add train_data/wine-quality.csv.dvc
- commit changes: git commit -m "train_data: remove 1000 rows"
- create a tag (v2) for later tracking of your data: git tag -a 'v2' -m 'version-2 wine-quality.csv removed 1000 rows'
- push the version of your data to dvc storage: dvc push
- verify push to dvc: ls -lR external_storage/

### Put Mlflow in action
- remove the dataset: rm -rf train_data/wine-quality.csv & rm -rf .dvc/cache
- reproduce dataset versions with dvc-api in: train.py
- train.py contains all mlflow tracking
- take a look in deploy-script.sh how to run mlflow server as a docker container
- go to mlflow-ui & look logged params, in my case: http://localhost:7755

#### Notes to spin-up a mlflow docker container
- create docker network: docker network create cesar_net
- run a postgress container: docker run --network cesar_net --expose=5432 -p 5432:5432 -d -v $PWD/pg_data_1/:/var/lib/postgresql/data/ --name pg_mlflow -e POSTGRES_USER='user_pg' -e POSTGRES_PASSWORD='pass_pg' postgres
- build Dockerfile: docker build -t mlflow_cesar .
- run mlflow server container: docker run -d -p 7755:5000 -v $PWD/artifacts:$PWD/artifacts --env-file local.env --network cesar_net --name test mlflow_cesar
