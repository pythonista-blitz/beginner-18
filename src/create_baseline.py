# import
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from prettytable import RANDOM
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.simplefilter("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# path
ROOT_PATH = Path.cwd().parent.resolve()
print("root path:", ROOT_PATH)
MLRUN_PATH = ROOT_PATH.parents[0] / "mlruns"
print("MLRUN path:", MLRUN_PATH)

# competition name(= experiment name)
EXPERIMENT_NAME = ROOT_PATH.name
print("experiment name:", EXPERIMENT_NAME)


# mlflow settings
mlflow.set_tracking_uri(str(MLRUN_PATH) + "/")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.start_run()
mlflow.xgboost.autolog()


# random seed
RANDOM_SEED = 126
mlflow.log_param(key="random_seed", value=RANDOM_SEED)


# load raw data
train = pd.read_csv(ROOT_PATH / "dataset" /
                    "train.csv")
submit = pd.read_csv(ROOT_PATH / "dataset" / "test.csv")
X = train.drop(["blueWins"], axis=1)
y = train["blueWins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=y
)

# 初心者コンペのせいか特に汚いデータが見られなかったのでクレンジングは割愛

# テスト用に分割
dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)


# 学習モデルの決定
model = XGBClassifier(
    n_estimators=10000,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    use_label_encoder=False,
)


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.0001, 0.1, log=True),
    }

    model.set_params(**params)

    cv_results = xgboost.cv(
        params,
        dtrain,
        num_boost_round=10000,
        seed=RANDOM_SEED,
        nfold=5,  # CVの分割数
        metrics=("logloss"),
        early_stopping_rounds=5,
        shuffle=True,
        stratified=True,
        verbose_eval=None,
    )

    return cv_results["test-logloss-mean"].min()


# パラメータチューニング
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
mlflow.log_metric(key="logloss", value=study.best_value)

# 最良のハイパーパラメータで学習
model_xgb = XGBClassifier(**study.best_params)
model_xgb.fit(X_train, y_train)

# 提出用テストデータ(test.csv)を推論
pred_xgb = model_xgb.predict_proba(submit)[:, 1]
pred_label = np.where(pred_xgb > 0.5, 1, 0)


# 提出用データの成形
submission = pd.DataFrame(
    {"gameId": submit["gameId"], "blueWins": pred_label})
submission.to_csv(ROOT_PATH / "submit" /
                  "submission_baseline.csv",
                  index=False,
                  header=False
                  )

mlflow.end_run()
