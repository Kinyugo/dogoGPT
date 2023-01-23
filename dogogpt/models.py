from typing import List, Optional, Tuple

import torch
from einops import rearrange
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType
from tqdm.autonotebook import trange

from dogogpt.modules import Block
from dogogpt.utils import eval_decorator


class Transformer(HyperparametersMixin, nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff_factor: int = 4,
        kernel_size: int = 5,
        dilation: int = 1,
        n_heads: int = 8,
        dropout: float = 0.0,
        has_attention: List[bool] = [False, False, False, False, True, True],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[
                Block(d_model, d_ff_factor, kernel_size, dilation, n_heads, dropout, x)
                for x in has_attention
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        inputs: TensorType["batch", "seq_len"],
        targets: Optional[TensorType["batch", "seq_len"]] = None,
    ) -> Tuple[TensorType["batch", "seq_len", "d_model"], Optional[TensorType[1]]]:
        x = self.embedding(inputs)
        x = self.blocks(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(rearrange(logits, "b l d -> b d l"), targets)

        return logits, loss

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        ids: TensorType["batch", "in_seq_len"],
        num_ids: int,
        context_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        num_parallel_ids: int = 1,
        verbose: bool = False,
        tag: str = "",
    ) -> TensorType["batch", "out_seq_len"]:
        assert not self.training, "Model is in training mode"
        for _ in trange(num_ids // num_parallel_ids, desc=tag, disable=(not verbose)):
            # Limit context to the maximum block size
            context = ids[..., -context_size:]
            # Compute the predictions
            logits, _ = self(context)
            # Focus only on the last timestep to predict the next
            logits = logits[:, -1, :]
            # Get probabilities
            logits = logits / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            next_ids = torch.multinomial(probs, num_samples=num_parallel_ids)
            # Append the sampled ids to the running sequence
            ids = torch.cat((ids, next_ids), dim=-1)

        return ids


if __name__ == "__main__":
    import string

    from dogogpt.dataloading import Tokenizer

    tkn = Tokenizer(string.printable)
    m = Transformer(tkn.vocab_size, 32)

    x = ["Hello World!", "World Hello?"]
    ids = tkn.batch_tokens_to_ids(x)
    ids = torch.stack(ids)
    s = m.generate(ids, 32, context_size=8, num_parallel_ids=2, verbose=True)
