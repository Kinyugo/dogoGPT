from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import MISSING


@dataclass
class CharacterDatamoduleConfig:
    src_path: str = MISSING
    num_chars: int = 512
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class TransformerConfig:
    d_model: int = 256
    d_ff_factor: int = 4
    kernel_size: int = 5
    dilation: int = 1
    n_heads: int = 8
    dropout: float = 0.0
    has_attention: List[bool] = field(
        default_factory=lambda: [False, False, False, False, True, True]
    )


@dataclass
class LitLanguageModelConfig:
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    sample_every_n_steps: int = 10
    num_samples: int = 4
    custom_seed: Optional[List[str]] = None
    num_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    num_parallel_ids: int = 1
    verbose: bool = False


@dataclass
class ModelCheckpointConfig:
    _target_: str = "lightning.pytorch.callbacks.ModelCheckpoint"
    dirpath: Optional[str] = None
    save_last: Optional[bool] = True
    verbose: bool = False
    mode: str = "min"
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TrainerConfig:
    _target_: str = "lightning.Trainer"
    accelerator: Optional[str] = "auto"
    accumulate_grad_batches: int = 1
    devices: Optional[Union[int, str]] = None
    default_root_dir: Optional[str] = None
    detect_anomaly: bool = False
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    limit_train_batches: Optional[Union[int, float]] = 1.0
    limit_val_batches: Optional[Union[int, float]] = 1.0
    limit_test_batches: Optional[Union[int, float]] = 1.0
    log_every_n_steps: int = 10
    precision: Union[int, str] = 32
    max_epochs: Optional[int] = 6
    max_steps: int = -1
    fast_dev_run: bool = False
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TrainingConfig:
    datamodule: CharacterDatamoduleConfig = field(
        default_factory=CharacterDatamoduleConfig
    )
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    lit_language_model: LitLanguageModelConfig = field(
        default_factory=LitLanguageModelConfig
    )
    model_checkpoint: ModelCheckpointConfig = field(
        default_factory=ModelCheckpointConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    seed: int = 0
    resume_ckpt_path: Optional[str] = None
    ckpt_path: str = "checkpoints/lm.pt"
    skip_training: bool = False


@dataclass
class SamplingConfig:
    ckpt_path: str = MISSING
    output_dir: str = MISSING

    seed_str: str = "\n"

    num_samples: int = 16
    batch_size: int = 4
    device: str = "cpu"
    dtype: str = "float32"

    num_tokens: int = 512
    context_size: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    num_parallel_tokens: int = 1
    verbose: bool = True
    tag: str = ""

    seed: int = 0
