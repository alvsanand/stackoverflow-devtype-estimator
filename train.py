import io
import requests
import zipfile
import os
import warnings
import sys
import logging as log
from functools import reduce

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


log.basicConfig(level=log.INFO)

_tmp_dir = '/tmp'
_dataset_url = (
    'https://drive.google.com'
    '/uc?export=download&id=1_9On2-nsBQIw3JiY43sWbrF8EjrqrR4U'
    )
_data_file = 'survey_results_public.csv'
_dataset_path = os.path.join(_tmp_dir, _data_file)
_dev_types_column = 'DevType'
_data_columns = ['Gender', 'RaceEthnicity', 'Age']
_dev_types = {
    'dbadm': 'Database administrator',
    'backdev': 'Back-end developer',
    'datasci': 'Data scientist or machine learning specialist',
    'manager': 'Product manager',
    'devops': 'DevOps specialist',
    'datanal': 'Data or business analyst',
}


def download_dataset():
    if not os.path.exists(_dataset_path):
        log.info("Downloading dataset")

        request = requests.get(_dataset_url)
        zipDocument = zipfile.ZipFile(io.BytesIO(request.content))
        zipDocument.extractall(_tmp_dir)

        log.info("Downloaded dataset")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    download_dataset()

    raw_data = pd.read_csv(_dataset_path)
    data = raw_data[_data_columns + [_dev_types_column]].dropna()

    log.info("Dataset size: {0}".format(raw_data.size))
    log.info("Filtered dataset size: {0}".format(data.size))

    dev_types = set(reduce(list.__add__,
                           map(lambda t: t.split(";"),
                               data["DevType"].unique().tolist())))

    for k, v in _dev_types.items():
        data[k] = data[_dev_types_column].str.contains(v)
        data[k] = data[k].map({True: 'YES', False: 'NO'})

    print(data)

    train, test = train_test_split(data)

    label_columns = list(_dev_types.keys())

    train_x = train.drop(_data_columns, axis=1)
    test_x = test.drop(label_columns, axis=1)
    train_y = train[label_columns]
    test_y = test[label_columns]

    max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_estimators = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_features = float(sys.argv[2]) if len(sys.argv) > 3 else 1

    with mlflow.start_run():
        lr = RandomForestClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    max_features=max_features)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        log.info("RandomForestClassifier model({0}, {1}, {2}):".format(
            max_depth, n_estimators, max_features))
        log.info("  RMSE: %s" % rmse)
        log.info("  MAE: %s" % mae)
        log.info("  R2: %s" % r2)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
