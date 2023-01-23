from typing import List, Optional, Tuple, Union

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, optim
from torchmetrics import Perplexity

from dogogpt.dataloading import Tokenizer
from dogogpt.models import Transformer


class LitLanguageModel(LightningModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        transformer: Transformer,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.999),
        sample_every_n_steps: int = 1000,
        num_samples: int = 4,
        custom_seed: Optional[List[str]] = None,
        num_tokens: int = 256,
        temperature: float = 1.0,
        num_parallel_ids: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["tokenizer", "transformer"])

        self.tokenizer = tokenizer
        self.transformer = transformer
        self.lr = lr
        self.betas = betas
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.custom_seed = custom_seed
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.num_parallel_ids = num_parallel_ids
        self.verbose = verbose

        self.train_perplexity = Perplexity()
        self.val_perplexity = Perplexity()
        self.test_perplexity = Perplexity()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.transformer(x)

    def training_step(
        self, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> Tensor:
        inputs, targets = batch

        logits, loss = self.transformer(inputs, targets)
        self.log("train_loss", loss)

        perplexity = self.train_perplexity(logits, targets)
        self.log("train_perplexity", perplexity)

        # Sample and log samples
        if self.global_step % self.sample_every_n_steps == 0:
            with torch.no_grad():
                # Log ground truth samples
                num_samples = min(self.num_samples, inputs.shape[0])
                self._log_text(inputs[:num_samples], "ground_truth")

                # Generate seed ids
                if self.custom_seed is not None:
                    ids = self.tokenizer.batch_tokens_to_ids(self.custom_seed)
                    ids = torch.stack(ids).to(self.device)
                else:
                    ids = torch.randint(
                        high=self.tokenizer.vocab_size,
                        size=(num_samples, 2),
                        device=self.device,
                    )

                # Generate and log samples without parallel sampling
                samples = self.transformer.generate(
                    ids,
                    self.num_tokens,
                    inputs.shape[-1],
                    self.temperature,
                    verbose=self.verbose,
                    tag="samples",
                )
                self._log_text(samples, "samples")

                # Generate and log samples with parallel sampling
                samples = self.transformer.generate(
                    ids,
                    self.num_tokens,
                    inputs.shape[-1],
                    self.temperature,
                    num_parallel_ids=self.num_parallel_ids,
                    verbose=self.verbose,
                    tag="samples_parallel",
                )
                self._log_text(samples, "samples_parallel")

                # Generate and Log samples from seed
                ids = inputs[..., : inputs.shape[-1] // 2]
                samples = self.transformer.generate(
                    ids,
                    self.num_tokens,
                    inputs.shape[-1],
                    self.temperature,
                    verbose=self.verbose,
                    tag="samples_seed",
                )
                self._log_text(samples, "samples_seeded")

        return loss

    def validation_step(
        self, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> Tensor:
        inputs, targets = batch

        logits, loss = self.transformer(inputs, targets)
        self.log("val_loss", loss)

        perplexity = self.val_perplexity(logits, targets)
        self.log("val_perplexity", perplexity)

    def test_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> Tensor:
        inputs, targets = batch

        logits, loss = self.transformer(inputs, targets)
        self.log("test_loss", loss)

        perplexity = self.test_perplexity(logits, targets)
        self.log("test_perplexity", perplexity)

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def _log_text(self, text: Tensor, tag: str) -> None:
        text = self.tokenizer.batch_ids_to_tokens(text)
        text = ["".join(x) for x in text]
        text = "\n---\n".join(text)

        self.logger.experiment.add_text(tag, text, global_step=self.global_step)
