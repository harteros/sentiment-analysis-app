import os
import sys
from logging import config

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from tqdm import tqdm

from ml.datasets.tweet_datamodule import TweetDataModule
from ml.utils.constants import EXPERIMENTS_DIR, ROOT_DIR
from ml.utils.helpers import load_model_at_version

tqdm.pandas(file=sys.stdout)


def test(model_name: str, version: str, model_type: str = "best-loss"):
    # Load a model from a specific version and checkpoint
    system = load_model_at_version(model_name, version, model_type)

    tweet_data_module = TweetDataModule()

    tb_logger = TensorBoardLogger(EXPERIMENTS_DIR, name=model_name, version=version)
    trainer = Trainer(logger=tb_logger)
    trainer.test(system, datamodule=tweet_data_module)
    predictions = trainer.predict(system, datamodule=tweet_data_module)

    results = pd.DataFrame(
        {'predictions': torch.cat(predictions).tolist(), 'labels': tweet_data_module.test_data["target"]}
    )
    print(classification_report(results.labels, results.predictions))


if __name__ == "__main__":
    config.fileConfig(os.path.join(ROOT_DIR, "logging.ini"))
    test(model_name="LSTMClassifier", version="version_0")
