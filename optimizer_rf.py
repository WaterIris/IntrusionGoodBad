import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path
import optuna
from optuna.trial import TrialState
from loguru import logger
import sys
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)

logger.add("optuna_trials.log",
           rotation="20 MB",
           level="DEBUG")

# logger.add(sys.stderr, level="DEBUG")


def load_data():
    dfs = []
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            if filename.endswith('.csv'):
                logger.info(f'reading {os.path.join(dirname, filename)}')
                dfs.append(pd.read_csv(os.path.join(dirname, filename)))
    df = pd.concat([dff for dff in dfs], ignore_index=True)
    del dfs
    logger.info('Done loading data')
    data_info = {
        'features': df.shape[1],
        'samples': df.shape[0],
        'memory GB': (int(df.memory_usage(index=False).iloc[0]) / 1000000000)
        * df.shape[1],
    }
    data_info_json = json.dumps(data_info, indent=4)
    logger.debug(f'Data info loaded:\n{data_info_json}')
    return df


def encode_label(data):
    encoder = LabelEncoder()
    data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
    return data


def get_data():
    data = load_data()
    data = encode_label(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.loc[:, (data != 0).any(axis=0)]
    data.drop(
        ['Bwd Packet Length Max', ' Max Packet Length', ' Idle Min'],
        axis=1,
        inplace=True,
    )
    data_info = {
        'features': data.shape[1],
        'samples': data.shape[0],
        'memory GB': (int(data.memory_usage(index=False).iloc[0]) / 1000000000)
        * data.shape[1],
    }
    data_info_json = json.dumps(data_info, indent=4)
    logger.debug(f'Data info cleaned:\n{data_info_json}')
    selector = SelectKBest(f_classif, k=7)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    logger.info('Selecting features')
    selector.fit(x, np.ravel(y.values))
    x_selected = selector.get_support(indices=True)
    x = x.iloc[:, x_selected]
    logger.info(f'Features selected {x.columns}')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    # Save current data split for retraining/reuse
    # save_path = "save"
    # df_train.to_csv(path_or_buf=f"{save_path}/train.csv")
    # df_test.to_csv(path_or_buf=f"{save_path}/test.csv")

    # scaler = MinMaxScaler().set_output(transform="pandas")
    logger.info('Scaling data')
    scaler = StandardScaler().set_output(transform='pandas')
    scaler = scaler.fit(df_train.iloc[:, :-1])

    # Save scaler for use after training models
    # joblib.dump(scaler, save_path/'scaler.gz')

    df_train.iloc[:, :-1] = scaler.transform(df_train.iloc[:, :-1])
    df_test.iloc[:, :-1] = scaler.transform(df_test.iloc[:, :-1])

    x_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1:]

    x_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1:]

    return x_train, y_train, x_test, y_test


def define_model(trial):
    logger.info(f"Starting Trial #{trial.number}")
    n_estimators = trial.suggest_int("n_estimators", 10, 40)
    max_depth = trial.suggest_int("max_depth", 50, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)

    model_RF = RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf)
    return model_RF


def objective(trial):
    session_name = f"learning_start_{trial.number}"
    save_path = Path(f"save/RandomForestClassifier/{session_name}/")
    logger.debug(f"Saving to {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = get_data()
    model = define_model(trial)
    model.fit(x_train, np.ravel(y_train.values))
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"RFC trial:{trial.number}")
    logger.info(f"Acc: {acc}, R2: {r2}")

    return r2


if __name__ == "__main__":
    logger.info("Starting optimizing process for RandomForestClassifier")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3,
                                         n_warmup_steps=15,
                                         interval_steps=5)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                study_name="RFC",
                                pruner=pruner,
                                load_if_exists=True)

    study.optimize(objective, n_trials=100)

    pruned = study.get_trials(deepcopy=False,
                              states=[TrialState.PRUNED])

    complete = study.get_trials(deepcopy=False,
                                states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"Total: {study.trials}, Pruned: {pruned}, Complete: {complete}")

    trial = study.best_trial
    logger.info(f"Best trial:{trial}")
    logger.info(f"Best trial Value: {trial.value}")
    logger.info("Parameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
