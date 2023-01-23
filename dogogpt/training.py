from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.callbacks import RichModelSummary, TQDMProgressBar
from omegaconf import OmegaConf

from dogogpt.configs import TrainingConfig
from dogogpt.dataloading import CharacterDataModule
from dogogpt.language_model import LanguageModel
from dogogpt.lit_models import LitLanguageModel
from dogogpt.models import Transformer
from dogogpt.utils import from_config


def run_training(config: TrainingConfig) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    L.seed_everything(config.seed)

    # -------------------------------------------
    # Data
    # -------------------------------------------
    datamodule = from_config(config.datamodule, target=CharacterDataModule)
    tokenizer = datamodule.tokenizer

    # -----------------------------------------
    # Model
    # ------------------------------------------
    transformer = from_config(
        config.transformer, target=Transformer, vocab_size=tokenizer.vocab_size
    )

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    lit_language_model = from_config(
        config.lit_language_model,
        target=LitLanguageModel,
        tokenizer=tokenizer,
        transformer=transformer,
    )

    # -----------------------------------------
    # Logger & Callbacks
    # ------------------------------------------
    model_checkpoint = from_config(config.model_checkpoint)
    callbacks = [model_checkpoint, RichModelSummary(), TQDMProgressBar()]

    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    trainer = from_config(config.trainer, callbacks=callbacks)

    # -----------------------------------------
    # Run Training
    # ------------------------------------------
    # Optionally run training
    if not config.skip_training:
        trainer.fit(
            lit_language_model, datamodule=datamodule, ckpt_path=config.resume_ckpt_path
        )

    # -----------------------------------------
    # Save Trained model
    # ------------------------------------------
    lm = LanguageModel(lit_language_model.tokenizer, lit_language_model.transformer)
    lm.save_pretrained(config.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", help="path to config file", type=str)
    args = parser.parse_args()

    default_training_config = OmegaConf.structured(TrainingConfig)
    file_training_config = OmegaConf.load(args.config_path)
    training_config = OmegaConf.merge(default_training_config, file_training_config)
    run_training(training_config)
