import os
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningCLI
from bert_classifier import BertClassifier
from bert_datamodule import BertDataModule


def cli_main():
    early_stopping = EarlyStopping(
        monitor="val_loss",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()
    cli = LightningCLI(
        BertClassifier,
        BertDataModule,
        run=False,
        save_config_callback=None,
        trainer_defaults={"callbacks": [early_stopping, checkpoint_callback, lr_logger]},
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
