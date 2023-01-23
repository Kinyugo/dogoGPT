import os
from argparse import ArgumentParser
from contextlib import nullcontext

import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.autonotebook import tqdm

from dogogpt.configs import SamplingConfig
from dogogpt.language_model import LanguageModel


def run_sampling(config: SamplingConfig) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    L.seed_everything(config.seed)

    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    ctx = (
        nullcontext()
        if config.device == "cpu"
        else torch.autocast(device_type=device, dtype=dtype)
    )
    os.makedirs(config.output_dir, exist_ok=True)

    # -----------------------------------------
    # Model
    # ------------------------------------------
    language_model = LanguageModel.from_pretrained(config.ckpt_path, device=device)

    # -----------------------------------------
    # Sampling
    # ------------------------------------------
    n_batches = config.num_samples // config.batch_size
    batches = np.array_split(range(config.num_samples), n_batches)

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Sampling", disable=(not config.verbose))
    ):
        with ctx:
            tokens = [config.seed_str] * len(batch)
            tokens = language_model(
                tokens,
                num_tokens=config.num_tokens,
                context_size=config.context_size,
                temperature=config.temperature,
                top_k=config.top_k,
                num_parallel_tokens=config.num_parallel_tokens,
                verbose=config.verbose,
                tag=config.tag,
            )

        # Save samples
        for i, item in enumerate(tokens):
            filename = f"{batch_idx * config.batch_size + i}.txt"
            with open(os.path.join(config.output_dir, filename), "w+") as f:
                f.write(item)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", help="path to config file", type=str)
    args = parser.parse_args()

    default_sampling_config = OmegaConf.structured(SamplingConfig)
    file_sampling_config = OmegaConf.load(args.config_path)
    sampling_config = OmegaConf.merge(default_sampling_config, file_sampling_config)
    run_sampling(sampling_config)
