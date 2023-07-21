import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy

from ml.utils.tokenizer import Tokenizer


class TweetSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.learning_rate = learning_rate

        self.model = model
        self.tokenizer = tokenizer

        self.criterion = nn.BCEWithLogitsLoss()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

    def on_fit_start(self):
        tb = self.logger.experiment

        epoch_loss = [r"train_loss_epoch", "val_loss_epoch"]
        epoch_accuracy = [r"train_acc_epoch", "val_acc_epoch"]

        layout = {
            "Epoch Metrics": {
                f"epoch_loss": ["Multiline", epoch_loss],
                "epoch_accuracy": ["Multiline", epoch_accuracy],
            },
        }
        tb.add_custom_scalars(layout)

    def forward(self, x: list[str]):
        input = self.tokenizer(x)
        input = tuple([i.to(self.device) for i in input])
        return self.model(input).view(-1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch["tweet"], batch["label"]

        pred = self(x)
        loss = self.criterion(pred, y.float())

        self.train_accuracy(pred, y)

        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        # this is the test loop
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch["tweet"], batch["label"]

        pred = self(x)
        loss = self.criterion(pred, y.float())

        if prefix == "val":
            self.val_accuracy(pred, y)
            self.log(f"{prefix}_acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        if prefix == "test":
            self.test_accuracy(pred, y)
            self.log(f"{prefix}_acc", self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch["tweet"], batch["label"]

        pred = self(x)
        # apply sigmoid function to prediction
        pred = F.sigmoid(pred)

        # convert to binary form with threshold at 0.5
        return (pred > 0.5).int()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
