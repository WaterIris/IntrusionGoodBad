import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)

DEVICE = torch.device('cpu')
BATCH_SIZE = 64
EPOCHS = 10


class CustomDataset(Dataset):
    def __init__(self, data):
        self.x = torch.from_numpy(data.iloc[:, :-1].values).to(torch.float64).unsqueeze(1).to(DEVICE)
        self.y = torch.from_numpy(data.iloc[:, -1:].values).to(torch.float64).to(DEVICE)
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1).to(torch.float64)
        self.bn1 = nn.BatchNorm1d(32).to(torch.float64)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1).to(torch.float64)
        self.bn2 = nn.BatchNorm1d(64).to(torch.float64)
        self.pool = nn.AdaptiveAvgPool1d(1).to(torch.float64)
        self.fc1 = nn.Linear(64, 64).to(torch.float64)
        self.fc2 = nn.Linear(64, num_classes).to(torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data():
    dfs = []
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            if filename.endswith('.csv'):
                print(f'reading {os.path.join(dirname, filename)}...')
                dfs.append(pd.read_csv(os.path.join(dirname, filename)))
    print('merging...')
    df = pd.concat([dff for dff in dfs], ignore_index=True)
    del dfs
    print('done')
    return df


def encode_label(data):
    encoder = LabelEncoder()
    data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
    return data


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%dT%H_%M")
    session_name = f"Learning_{start_time}"
    save_path = Path(f"save/{session_name}/")
    save_path.mkdir(parents=True, exist_ok=True)

    data = load_data()
    data = encode_label(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    selector = SelectKBest(f_classif, k=5)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    selector.fit(x, y)
    x_selected = selector.get_support(indices=True)
    x = x.iloc[:, x_selected]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        shuffle=True)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    # Save current data split for retraining/reuse
    # save_path = "save"
    # df_train.to_csv(path_or_buf=f"{save_path}/train.csv")
    # df_test.to_csv(path_or_buf=f"{save_path}/test.csv")

    # scaler = MinMaxScaler().set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    scaler = scaler.fit(df_train.iloc[:, :-1])

    # Save scaler for use after training models
    # joblib.dump(scaler, save_path/'scaler.gz')

    df_train.iloc[:, :-1] = scaler.transform(df_train.iloc[:, :-1])
    df_test.iloc[:, :-1] = scaler.transform(df_test.iloc[:, :-1])

    x_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1:]

    x_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1:]

    # RandomForestClassifier
    model_RF = RandomForestClassifier(n_estimators=10,
                                      max_depth=5)

    model_RF.fit(x_train, np.ravel(y_train.values))
    y_pred = model_RF.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("-" * 120)
    print("RandomForestClassifier :")
    print(f"Acc score: {acc}")
    print(f"R2 score: {r2}")

    # CNN
    dataset_train = CustomDataset(data=df_train)
    dataset_test = CustomDataset(data=df_test)

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    loader_test = DataLoader(dataset=dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    n_categories = len(np.unique(y))
    model_CNN = CNN1D(n_categories)
    model_CNN.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_CNN.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch}]")
        model_CNN.train()
        current_epoch_loss = 0
        for x_batch, y_batch in loader_train:
            optimizer.zero_grad()
            output = model_CNN(x_batch)
            print(output)
            print(y_batch)
            exit()
            loss = criterion(torch.flatten(output), torch.flatten(y_batch))
            loss.backward()
            optimizer.step()
            current_epoch_loss += loss.item()

        avg_loss = current_epoch_loss / len(loader_train)
