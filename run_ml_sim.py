import os
import glob

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from helper_scripts.ml_helpers import process_data, plot_confusion
from helper_scripts.ml_helpers import save_model


def _train_test_knn(df_processed: pd.DataFrame, sim_dict: dict, erlang: str):
    predictor_df = df_processed['num_segments']
    feature_df = df_processed.drop('num_segments', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df,
                                                        test_size=sim_dict['test_size'], random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    plot_confusion(sim_dict=sim_dict, y_test=y_test, y_pred=y_pred, erlang=erlang, algorithm='KNN')

    save_model(sim_dict=sim_dict, model=knn, algorithm='knn', erlang=erlang)


def _train_test_dt(df_processed: pd.DataFrame, sim_dict: dict, erlang: str):
    predictor_df = df_processed['num_segments']
    feature_df = df_processed.drop('num_segments', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df, test_size=sim_dict['test_size'],
                                                        random_state=42)
    dt_obj = DecisionTreeClassifier(random_state=0)
    dt_obj.fit(x_train, y_train)
    y_pred = dt_obj.predict(x_test)
    plot_confusion(sim_dict=sim_dict, y_test=y_test, y_pred=y_pred, erlang=erlang, algorithm='Decision Tree')

    save_model(sim_dict=sim_dict, model=dt_obj, algorithm='decision_tree', erlang=erlang)


def _train_test_lr(df_processed: pd.DataFrame, sim_dict: dict, erlang: str):
    predictor_df = df_processed['num_segments']
    feature_df = df_processed.drop('num_segments', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df, test_size=sim_dict['test_size'],
                                                        random_state=42)
    lr_obj = LogisticRegression(random_state=0, n_jobs=-1)
    lr_obj.fit(x_train, y_train)
    y_pred = lr_obj.predict(x_test)
    plot_confusion(sim_dict=sim_dict, y_test=y_test, y_pred=y_pred, erlang=erlang,
                   algorithm='Logistic Regression')

    save_model(sim_dict=sim_dict, model=lr_obj, algorithm='logistic_regression', erlang=erlang)


def extract_value(path: str):
    """
    Extracts the erlang value from a path.

    :param path: Input path with the embedded value.
    :return: The embedded value.
    :rtype: str
    """
    parts = path.split('/')
    filename = parts[-1]
    filename_parts = filename.split('_')
    value = filename_parts[0]
    if '.' in value:
        value = value.split('.')[0]
    return value


def _handle_training(sim_dict: dict, file_path: str):
    data_frame = pd.read_csv(file_path)

    erlang = extract_value(path=file_path)
    df_processed = process_data(sim_dict=sim_dict, input_df=data_frame, erlang=erlang)
    if sim_dict['ml_model'] == 'knn':
        _train_test_knn(df_processed=df_processed, sim_dict=sim_dict, erlang=erlang)
    elif sim_dict['ml_model'] == 'logistic_regression':
        _train_test_lr(df_processed=df_processed, sim_dict=sim_dict, erlang=erlang)
    elif sim_dict['ml_model'] == 'decision_tree':
        _train_test_dt(df_processed=df_processed, sim_dict=sim_dict, erlang=erlang)
    else:
        raise NotImplementedError


def _run(sim_dict: dict):
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    # TODO: Only support for running one process (s1)
    sim_dict = sim_dict['s1']
    base_fp = 'data/output/'

    train_dir = os.path.join(base_fp, sim_dict['train_file_path'])
    train_files = glob.glob(os.path.join(train_dir, "*.csv"))

    for train_fp in train_files:
        _handle_training(sim_dict=sim_dict, file_path=train_fp)


def _setup_ml_sim():
    """
    Gets the simulation input parameters.

    :return: The simulation input parameters.
    :rtype: dict
    """
    args_dict = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_dict=args_dict, config_path=config_path)

    return sim_dict


def run_ml_sim():
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    sim_dict = _setup_ml_sim()
    _run(sim_dict=sim_dict)


if __name__ == '__main__':
    run_ml_sim()
