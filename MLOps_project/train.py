import hydra
import lightning.pytorch as pl
import torch
from conf.config import Config
from data import MyDataModule
from hydra.core.config_store import ConfigStore
from model import CNN_new
from utils import convert_to_onnx, save_all


cs = ConfigStore.instance()
cs.store(name="config", group="first", node=Config)


@hydra.main(config_path="./../conf", config_name="config", version_base="1.3")
def main(cfg: Config):
    print("start")
    # pl.seed_everithing(32)
    torch.set_float32_matmul_precision("medium")

    dm = MyDataModule(cfg=cfg)

    model = CNN_new(conf=cfg.model)
    print("\n\nMODEL IS SETTED\n\n")
    loggers = [
        # pl.loggers.CSVLogger("./.logs/my-csv-logs", name="first_experiment"),
        pl.loggers.MLFlowLogger(
            # experiment_id = cfg.loggers.mlflow.experiment_name,
            experiment_name=cfg.loggers.mlflow.experiment_name,  # cfg.artifacts.experiment_name,
            tracking_uri=cfg.loggers.mlflow.tracking_uri,
            artifact_location="./../conf/config.yaml",
        )
    ]
    loggers[0].log_hyperparams(cfg)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(),
    ]

    trainer = pl.Trainer(
        log_every_n_steps=cfg.training.log_every_n_steps,
        max_epochs=cfg.training.epochs,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)

    print("finish training\nmodel saving")
    save_all(
        model,
        cfg.model,
        save_path=cfg.model.save_path,
        save_name=cfg.model.save_name,
    )

    if cfg.model.onnx_parameters.export_to_onnx:
        convert_to_onnx(model=model, conf=cfg.model.onnx_parameters)

    print(f"model saved at {cfg.model.save_name}")


if __name__ == "__main__":
    main()
