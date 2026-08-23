# Project repository: https://github.com/2022jiangjiazheng

import os
import time
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_processing.greatwall_data import GreatwallDataModule
from models.sggwseg_segmentation import SGGWSeg


def parse_devices(value):
    """Convert device indices to the format expected by Lightning 2.x."""
    if value == "auto":
        return "auto"
    return [int(device.strip()) for device in value.split(",")]


def main(hparams, run_number, running_mode, checkpoint_path):
    checkpoint_dir = os.path.join(hparams.project_dir, "checkpoints", f"run_{run_number}")
    tb_logs_dir = os.path.join(hparams.project_dir, "tb_logs", f"run_{run_number}")
    temporary_checkpoint = os.path.join(checkpoint_dir, "temporary.ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="avg_metric_validation",
        dirpath=checkpoint_dir,
        filename="-{epoch:02d}-{avg_metric_validation:.2f}",
        mode="max",
        save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="avg_metric_validation",
        patience=30,
        verbose=False,
        mode="max",
        check_finite=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(tb_logs_dir, name="log", default_hp_metric=False)

    devices = parse_devices(hparams.devices)
    number_of_devices = len(devices) if isinstance(devices, list) else 1
    strategy = hparams.strategy
    if strategy == "auto" and number_of_devices > 1:
        strategy = "ddp"

    callbacks = [checkpoint_callback, lr_monitor]
    if running_mode == "training":
        callbacks.insert(0, early_stop_callback)

    trainer_arguments = dict(
        accelerator=hparams.accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        # CUDA NLLLoss2d in PyTorch 2.1 has no deterministic implementation.
        deterministic=False,
        gradient_clip_val=1.0,
        logger=logger,
        max_epochs=hparams.epochs,
        log_every_n_steps=hparams.log_every_n_steps,
        sync_batchnorm=number_of_devices > 1,
    )

    if running_mode == "batch_overfit":
        trainer_arguments.update(max_epochs=1000, overfit_batches=1)
    elif running_mode == "debugging":
        trainer_arguments.update(fast_dev_run=3)

    trainer = Trainer(**trainer_arguments)

    datamodule = GreatwallDataModule(
        batch_size=hparams.batch_size,
        augmentation=running_mode != "batch_overfit",
        data_dir=hparams.data_dir,
        project_dir=hparams.project_dir,
        bright=hparams.bright,
        wrap=hparams.wrap,
        noise=hparams.noise,
        rotate=hparams.rotate,
        hflip=hparams.hflip,
        vflip=hparams.vflip,
    )
    model = SGGWSeg(vars(hparams))
    print(model)

    # Lightning 2.x resumes training through fit(ckpt_path=...).
    resume_path = checkpoint_path
    if resume_path is None and os.path.isfile(temporary_checkpoint):
        resume_path = temporary_checkpoint
    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)

    if running_mode == "training":
        trainer.save_checkpoint(temporary_checkpoint)


def build_parser():
    parser = ArgumentParser(description="Train SGGWSeg on the Great Wall dataset.")
    parser.add_argument("--project_dir", default=".", help="Project root directory.")
    parser.add_argument("--data_dir", default="data", help="Preprocessed dataset directory.")
    parser.add_argument(
        "--training_mode",
        choices=("training", "debugging", "batch_overfit"),
        default="training",
        help="Training, quick debugging, or single-batch overfitting mode.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of training epochs.")
    parser.add_argument("--run_number", type=int, default=6, help="Run identifier used by logs and checkpoints.")
    parser.add_argument("--checkpoint_path", default=None, help="Checkpoint used to resume training.")
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument(
        "--devices",
        default="0",
        help="GPU indices separated by commas, for example '0' or '0,1,2,3'.",
    )
    parser.add_argument("--strategy", default="auto", help="Lightning strategy, e.g. auto or ddp.")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    return SGGWSeg.add_model_specific_args(parser)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    arguments = build_parser().parse_args()
    print(vars(arguments))

    start_time = time.time()
    main(
        arguments,
        arguments.run_number,
        arguments.training_mode,
        arguments.checkpoint_path,
    )
    print(f"Total time for training: {(time.time() - start_time) / 3600:.2f} h")
