import os
from typing import List, Optional

import torch
from torch import nn

from dogogpt.dataloading import Tokenizer
from dogogpt.models import Transformer
from dogogpt.utils import eval_decorator


class LanguageModel(nn.Module):
    def __init__(self, tokenizer: Tokenizer, transformer: Transformer) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.transformer = transformer

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @torch.no_grad()
    @eval_decorator
    def forward(
        self,
        tokens: List[str],
        num_tokens: int,
        context_size: int,
        temperature: float = 1.0,
        num_parallel_ids: int = 1,
        verbose: bool = False,
        tag: str = "",
    ) -> List[str]:
        # Switch to evaluation mode
        self.eval()

        # Convert raw tokens into ids
        ids = self.tokenizer.batch_tokens_to_ids(tokens)
        ids = torch.stack(ids).to(self.device)

        # Generate new ids
        ids = self.transformer.generate(
            ids, num_tokens, context_size, temperature, num_parallel_ids, verbose, tag
        )

        # Convert ids to tokens
        tokens = self.tokenizer.batch_ids_to_tokens(ids)
        tokens = ["".join(x) for x in tokens]

        return tokens

    @torch.no_grad()
    def save_pretrained(self, ckpt_path: str) -> None:
        checkpoint = {
            "vocab": self.tokenizer.vocab,
            "transformer_hparams": self.transformer.hparams,
            "transformer_state_dict": self.transformer.state_dict(),
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(checkpoint, ckpt_path)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        # Load checkpoint from disk
        checkpoint = torch.load(ckpt_path)

        # Load states of various models from the checkpoint
        tokenizer = Tokenizer(checkpoint["vocab"])
        transformer = Transformer(**checkpoint["transformer_hparams"])
        transformer.load_state_dict(checkpoint["transformer_state_dict"])

        return cls(tokenizer, transformer).to(device)
