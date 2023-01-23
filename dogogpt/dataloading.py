from typing import List, Optional, Sequence, Union

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class Tokenizer:
    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = vocab

        self._str_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self._idx_to_str = {i: ch for ch, i in self._str_to_idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __call__(self, tokens: str) -> Tensor:
        return self.tokens_to_ids(tokens)

    def batch_tokens_to_ids(self, batch_tokens: List[Sequence[str]]) -> List[Tensor]:
        return [self.tokens_to_ids(tokens) for tokens in batch_tokens]

    def batch_ids_to_tokens(
        self, batch_indices: Union[Tensor, List[Tensor]]
    ) -> List[List[str]]:
        return [self.ids_to_tokens(indices) for indices in batch_indices]

    def tokens_to_ids(self, tokens: Sequence[str]) -> Tensor:
        return torch.tensor([self._str_to_idx.get(t) for t in tokens])

    def ids_to_tokens(self, indices: Tensor) -> List[str]:
        return [self._idx_to_str.get(idx) for idx in indices.tolist()]


class CharacterDataset(Dataset):
    def __init__(self, src_path: str, num_chars: int) -> None:
        super().__init__()

        self.src_path = src_path
        self.num_chars = num_chars

        self._text = self.__load_text()
        self._vocab = self.__build_vocab()
        self._tokenizer = Tokenizer(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def __len__(self) -> int:
        return len(self._text) - (self.num_chars + 1)

    def __getitem__(self, index: int) -> Tensor:
        subtext = self._text[index : index + self.num_chars + 1]
        input_ids = self._tokenizer(subtext[:-1])
        target_ids = self._tokenizer(subtext[1:])

        return input_ids, target_ids

    def __load_text(self) -> str:
        with open(self.src_path, "rb") as fh:
            return fh.read().decode(encoding="utf-8")

    def __build_vocab(self) -> List[str]:
        return sorted(set(self._text))


class CharacterDataModule(L.LightningDataModule):
    def __init__(
        self,
        src_path: str,
        num_chars: int,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ) -> None:
        super().__init__()

        self.dataset = CharacterDataset(src_path, num_chars)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split

    @property
    def tokenizer(self) -> Tokenizer:
        return self.dataset._tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        train_split = 1.0 - (self.val_split + self.test_split)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_split, self.val_split, self.test_split]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
