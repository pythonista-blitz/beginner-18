
from operator import index
from pathlib import Path
from tokenize import group

import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

ROOT_PATH = Path(".").resolve().parent

# read data


def read_dataset():
    train = pd.read_csv(ROOT_PATH / "dataset" / "train.csv")
    test = pd.read_csv(ROOT_PATH / "dataset" / "test.csv")
    return train, test


# if __name__ == "__main__":
train, test = read_dataset()
train.head()


train["blueTotalGold"].hist()

# 経験値とGoldは試合時間が経つにつれ増加する
# 試合時間という概念を予測する


train.groupby("blueWins")["blueTotalGold"].plot.hist(
    bins=20, alpha=0.5, legend=True)


train.groupby("blueWins")["blueTotalExperience"].plot.hist(
    bins=20, alpha=0.5, legend=True)


train.groupby("blueWins")["blueEliteMonsters"].plot.hist(
    bins=20, alpha=0.5, legend=True)


train.groupby("blueWins")["blueDragons"].plot.hist(
    bins=20, alpha=0.5, legend=True)


train.groupby(["blueWins"]).count()


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

columns = ["feature", "mean",
           "std", "skewness", "kurtosis"]
df = pd.DataFrame(columns=columns)
# for col in train.columns:
#     row = pd.Series(
#         [
#             col,
#             round(train[col].mean(), 2),
#             round(train[col].std(), 2),
#             round(train[col].skew(), 2),
#             round(train[col].kurt(), 2)
#         ]
#     )
#     df = df.append(row, ignore_index=True)

# df.to_csv(ROOT_PATH / "submit" / "statics.csv", index=False)
for col in train.columns:
    plt.figure()
    plt.title(col)
    train.groupby("blueWins")[col].plot.hist(
        bins=20, alpha=0.5, legend=True)
    plt.savefig(ROOT_PATH/"submit"/f"{col}.png")
    plt.show()

nr_rows = 3  # 図を表示する際の行数
nr_cols = 3  # 図を表示する際の列数
# nr_rows * nr_colsがカラム数を超えるように設定。

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5, nr_rows*3))

li_num_feats = list(train.columns)
li_not_plot = ["gameId", "blueFirstBlood"]
li_plot_num_feats = [c for c in li_num_feats if c not in li_not_plot]
target = "blueWins"

for r in range(0, nr_rows):
    for c in range(0, nr_cols):
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(train[li_plot_num_feats[i]],
                        train[target], ax=axs[r][c])
            stp = stats.pearsonr(train[li_plot_num_feats[i]], train[target])
            str_title = "r = " + \
                "{0:.2f}".format(stp[0]) + "      " "p = " + \
                "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title, fontsize=11)

plt.tight_layout()
plt.show()
plt.savefig(ROOT_PATH/"submit"/"outlier.png")
