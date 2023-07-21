import os
import sys
from logging import config

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchtext.vocab import GloVe
from tqdm import tqdm

from ml.datasets.tweet_datamodule import TweetDataModule
from ml.engines.system import TweetSystem
from ml.models.lstm import LSTMClassifier
from ml.utils.constants import EXPERIMENTS_DIR, ROOT_DIR, EMBEDDINGS_DIR
from ml.utils.tokenizer import Tokenizer

tqdm.pandas(file=sys.stdout)


def get_callbacks():
    tqdm_callback = TQDMProgressBar(refresh_rate=1)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        filename="best-loss-model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last-model-{epoch:02d}-{val_loss:.2f}"
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)

    return [tqdm_callback, checkpoint_callback, early_stopping_callback]


def train():
    # Setup the data module
    # This will take care of downloading the data, preprocessing it,
    # splitting it into train/val/test sets, and encoding the labels
    tweet_data_module = TweetDataModule()
    tweet_data_module.prepare_data()
    tweet_data_module.setup(stage="fit")

    pretrained_embeddings = GloVe(cache=EMBEDDINGS_DIR, name='twitter.27B', dim=100)

    # Create the Tokenizer vocab based on the pretrained embeddings and the training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts_and_embeddings(tweet_data_module.train_data.text, pretrained_embeddings)
    embeddings_matrix = tokenizer.get_embeddings_matrix(pretrained_embeddings)

    model = LSTMClassifier(embeddings_matrix)
    tweet_system = TweetSystem(model=model, tokenizer=tokenizer)

    tb_logger = TensorBoardLogger(EXPERIMENTS_DIR, name=model.name)
    output_dir = os.path.join(EXPERIMENTS_DIR, model.name, f"version_{tb_logger.version}")
    tokenizer.save(output_dir)

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=10,
        callbacks=get_callbacks(),
        logger=tb_logger,
        log_every_n_steps=1,
    )

    trainer.fit(model=tweet_system, datamodule=tweet_data_module)


if __name__ == "__main__":
    config.fileConfig(os.path.join(ROOT_DIR, "logging.ini"))
    train()
