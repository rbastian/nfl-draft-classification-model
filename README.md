# nfl-draft-classification-model

## Training

Training data is expected to be in the following shape: 

| NAME | POWERFIVE | SOPH | JR | SR | HEIGHT | WEIGHT | 40-YD | AGE | G | COMBINE | BRUGLER |
| ---- | --------- | ---- | -- | -- | ------ | ------ | ----- | --- | - | ------- | ------- |
| STR  | BOOL      | BOOL | BOOL | BOOL | INT | INT | FLOAT | FLOAT | INT | BOOL | BOOL |
| Bryce Young | 1 | 0 | 1 | 0 | 5101 | 204 | 4.52 | 21.76 | 34 | 1 | 1 | | 

Training data is expected to be a /data subdirectory from the project root and in .csv format.

train.py leverages MLFlow to register experiments and runs. 

## Prediction

predict.py leverages MLFlow to pull models from the MLFlow repository by run_id.

