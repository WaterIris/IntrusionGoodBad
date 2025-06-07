from pathlib import Path
import os
import loguru
import json
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from modules.cnn_model import Cnnid
from modules.custom_dataset import IntrusionDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

class Comparator:
    def __init__(self, data_dir, save_dir, n_feature_selected):
        self._data_dir = Path(data_dir)
        self._save_dir = Path(save_dir)
        self._logger = self._get_logger(save_dir)
        self._n_best_features = n_feature_selected
        self._data = self._load_data()
        self.current_data_inf = self._get_data_inf()
        self._data_train = None
        self._data_test = None
        self._score = {}

    def _load_data(self):
        self._logger.info("Loading data")
        start_loading_t = time.time()
        dfs = []
        for dirname, _, filenames in os.walk('data'):
            for filename in filenames:
                if filename.endswith('.csv'):
                    self._logger.info(f'Reading {os.path.join(dirname, filename)}')
                    dfs.append(pd.read_csv(os.path.join(dirname, filename)))
        df = pd.concat([dff for dff in dfs], ignore_index=True)
        del dfs
        end_loading_t = time.time()
        self._logger.info(f"Time taken {end_loading_t - start_loading_t}")
        return df

    @staticmethod
    def _get_logger(save_dir):
        logger = loguru.logger
        logger.add(Path(f"{save_dir}/comparison.log"),
                   rotation="20 MB",
                   level="DEBUG")
        return logger

    def _get_data_inf(self):
        data_info = {'features': self._data.shape[1],
                     'samples': self._data.shape[0],
                     'memory GB': (int(self._data.memory_usage(index=False).iloc[0]) / 10 ** 9) * self._data.shape[1]}
        data_info_json = json.dumps(data_info, indent=4)
        self._logger.info(f'\n{data_info_json}')
        return None

    @staticmethod
    def _encode_labels(data):
        encoder = LabelEncoder()
        data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
        return data

    def _scale_data(self):
        self._logger.info('Scaling data')
        scaler = StandardScaler().set_output(transform='pandas')
        scaler = scaler.fit(self._data_train.iloc[:, :-1])

        # Save scaler for use after training models
        joblib.dump(scaler, self._save_dir/'scaler.gz')

        self._data_train.iloc[:, :-1] = scaler.transform(self._data_train.iloc[:, :-1])
        self._data_test.iloc[:, :-1] = scaler.transform(self._data_test.iloc[:, :-1])
        return None

    def _prepare_data(self):
        self._logger.info("Starting data preparation")
        self._logger.info('Data before cleaning')
        self._get_data_inf()
        data = self._encode_labels(self._data)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data = data.loc[:, (data != 0).any(axis=0)]
        data.drop(['Bwd Packet Length Max', ' Max Packet Length', ' Idle Min'],
                  axis=1,
                  inplace=True)
        self._logger.info("Cleaned data information")
        self._data = data
        self._get_data_inf()
        self._logger.info(f'Selecting {self._n_best_features} features')
        selector = SelectKBest(f_classif, k=self._n_best_features)
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1:]

        selector.fit(x, np.ravel(y.values))
        x_selected = selector.get_support(indices=True)
        x = x.iloc[:, x_selected]
        self._logger.info(f'Features selected {list(x.columns)}')
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            shuffle=True)
        self._logger.info("Creating test, train dataframes")
        df_train = pd.concat([x_train, y_train], axis=1)
        df_test = pd.concat([x_test, y_test], axis=1)

        self._logger.info(f"Saving dataframes in {self._save_dir}")
        df_train.to_csv(path_or_buf=f"{self._save_dir}/train.csv")
        df_test.to_csv(path_or_buf=f"{self._save_dir}/test.csv")

        self._data_train = df_train
        self._data_test = df_test

        self._scale_data()

        return None

    def _get_train_test(self):
        x_train = self._data_train.iloc[:, :-1]
        y_train = self._data_train.iloc[:, -1:]

        x_test = self._data_test.iloc[:, :-1]
        y_test = self._data_test.iloc[:, -1:]

        return x_train, y_train, x_test, y_test

    def _score_model(self, model_type, y_real, y_pred):
        acc = accuracy_score(y_real, y_pred)
        f1 = f1_score(y_real, y_pred)
        scores_info = {model_type: {"Acc": acc, "F1": f1}}
        self._logger.info(f"Scoring {model_type} Done")
        self._score.update(scores_info)
        return None

    def _test_RFC(self):
        self._logger.info("Starting RandomForestClassifier training")
        x_train, y_train, x_test, y_test = self._get_train_test()
        model_RF = RandomForestClassifier(n_estimators=10, max_depth=10)
        self._logger.info("Fitting model")
        model_RF.fit(x_train, np.ravel(y_train.values))
        y_pred = model_RF.predict(x_test)
        self._logger.info("Starting RandomForestClassifier test")
        self._logger.info(f"\n{classification_report(y_test, y_pred)}")
        return None

    def _test_CNN(self):
        self._logger.info("Starting ConvolutionallNeuralNetwork training")
        x_train, y_train, x_test, y_test = self._get_train_test()
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy().ravel()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy().ravel()
        # Konwersja do tensora
        X_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = IntrusionDataset(X_train_tensor, y_train_tensor)
        test_dataset = IntrusionDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        num_classes = len(np.unique(y_train))
        model = Cnnid(num_classes)

        device = torch.device('cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(20):
            self._logger.info(f"Epoch {epoch + 1}")
            # === Trening ===
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # avg_loss = total_loss / len(train_loader)

            # === Ewaluacja ===
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y_batch.numpy())

        self._logger.info(f"\n{classification_report(all_targets, all_preds)}")

        return None

    def start_comparing(self):
        self._prepare_data()
        self._test_RFC()
        self._test_CNN()
        pass


if __name__ == "__main__":
    data_dir = "data"
    save_dir = "save"
    comparer = Comparator(data_dir=data_dir,
                          save_dir=save_dir,
                          n_feature_selected=7)
    comparer.start_comparing()
