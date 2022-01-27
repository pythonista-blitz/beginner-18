# %%
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt

ROOT_PATH = Path(".").resolve().parent
# %%
# read data


def read_dataset():
    train = pd.read_csv(ROOT_PATH / "dataset" / "train.csv")
    test = pd.read_csv(ROOT_PATH / "dataset" / "test.csv")
    return train, test


# %%
# if __name__ == "__main__":
train, test = read_dataset()
train.head()

# %%
train["blueTotalGold"].hist()

# 経験値とGoldは試合時間が経つにつれ増加する
# 試合時間という概念を予測する


# %%
train.groupby("blueWins")["blueTotalGold"].plot.hist(
    bins=20, alpha=0.5, legend=True)

# %%
train.groupby("blueWins")["blueTotalExperience"].plot.hist(
    bins=20, alpha=0.5, legend=True)

# %%
train.groupby("blueWins")["blueEliteMonsters"].plot.hist(
    bins=20, alpha=0.5, legend=True)

# %%
train.groupby("blueWins")["blueDragons"].plot.hist(
    bins=20, alpha=0.5, legend=True)

# %%
train.groupby(["blueWins"]).count()

# %%
