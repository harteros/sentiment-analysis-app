import logging
import os

import pandas as pd
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ml.data.make_dataset import make_dataset
from ml.data.preprocessing import preprocess
from ml.datasets.tweet_dataset import TweetDataset
from ml.utils.constants import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class TweetDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, split: float = 0.1, recreate_data: bool = False):
        super().__init__()

        self.save_hyperparameters("batch_size")

        self.batch_size = batch_size
        self.split = split
        self.recreate_data = recreate_data
        self.has_setup = False

    def prepare_data(self):
        if self.recreate_data or not os.path.exists(PROCESSED_DATA_DIR) or os.listdir(PROCESSED_DATA_DIR) == 0:
            make_dataset()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders

        if stage == "fit":
            if not self.has_setup:
                logger.info("Loading train and val datasets")
                self.has_setup = True

                data = self.process_data(stage)
                # split data
                self.train_data, self.val_data = train_test_split(data, test_size=self.split, random_state=0)

                self.train_dataset = TweetDataset(self.train_data)
                self.val_dataset = TweetDataset(self.val_data)

        if stage in (None, "test", "predict"):
            logger.info("Loading test dataset")

            self.test_data = self.process_data(stage)
            self.test_dataset = TweetDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def process_data(self, stage: str):
        file_name = None

        if stage == "fit":
            file_name = "train.csv"

        if stage in (None, "test", "predict"):
            file_name = "test.csv"

        if file_name is None:
            raise ValueError(f"Stage {stage} is not valid.")

        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file_name), encoding="ISO-8859-1")
        data["text"] = data.text.progress_apply(lambda x: preprocess(x))
        return data
