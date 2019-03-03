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
_cols = {
    'dev_type': 'DevType',
    'gender': 'Gender',
    'race': 'RaceEthnicity',
    'age': 'Age',
}
_dev_types = {
    'dbadm': 'Database administrator',
    'backdev': 'Back-end developer',
    'datasci': 'Data scientist or machine learning specialist',
    'manager': 'Product manager',
    'devops': 'DevOps specialist',
    'datanal': 'Data or business analyst',
}
_genders = {
    'Female': 0,
    'Male': 1,
}
_races = {
    'Black or of African descent': 0,
    'East Asian': 1,
    'Hispanic or Latino/Latina': 2,
    'Middle Eastern': 3,
    'Native American, Pacific Islander, or Indigenous Australian': 4,
    'South Asian': 5,
    'White or of European descent': 6,
}
_ages = {
    'Under 18 years old': 0,
    '18 - 24 years old': 1,
    '25 - 34 years old': 2,
    '35 - 44 years old': 3,
    '45 - 54 years old': 4,
    '55 - 64 years old': 5,
    '65 years or older': 6,
}


def print_values(data):
    for c in data:
        values = set(reduce(list.__add__,
                            map(lambda t: str(t).split(";"),
                                data[c].unique().tolist())))
        log.info("{0} -> {1}".format(c, values))


def download_dataset():
    if not os.path.exists(_dataset_path):
        log.info("Downloading dataset")

        request = requests.get(_dataset_url)
        zipDocument = zipfile.ZipFile(io.BytesIO(request.content))
        zipDocument.extractall(_tmp_dir)

        log.info("Downloaded dataset")


def filter_values(data):
    return data[_cols.values()][
        data[_cols['gender']].isin(['Male', 'Female'])][
        data[_cols['race']].isin(_races.keys())].dropna()


def map_values(data):
    data = data.replace({
                        _cols['gender']: _genders,
                        _cols['race']: _races,
                        _cols['age']: _ages,
                        })

    for k, v in _dev_types.items():
        data[k] = data[_cols['dev_type']].str.contains(v)
        data[k] = data[k].map({True: 1, False: 0})

    return data.drop(_cols['dev_type'], axis=1)


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
    log.info("Dataset size: {0}".format(raw_data.size))
    # log.info(raw_data)

    data = map_values(filter_values(raw_data))
    log.info("Filtered dataset size: {0}".format(data.size))

    train, test = train_test_split(data)

    label_cols = list(_dev_types.keys())

    train_x = train.drop(label_cols, axis=1)
    test_x = test.drop(label_cols, axis=1)
    train_y = train[label_cols]
    test_y = test[label_cols]

    max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_estimators = float(sys.argv[2]) if len(sys.argv) > 2 else 1000

    with mlflow.start_run():
        lr = RandomForestClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        log.info("RandomForestClassifier model({0}, {1}):".format(
            max_depth, n_estimators))
        log.info("  RMSE: %s" % rmse)
        log.info("  MAE: %s" % mae)
        log.info("  R2: %s" % r2)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
