"""Configuration management module for the Nari model.

This module provides comprehensive configuration management for the Nari model,
including model architecture settings, training parameters, and checkpoint handling.
It uses Pydantic for robust validation and type checking of all configuration parameters.

Key components:
- Model configuration (architecture, dimensions, vocabulary)
- Training hyperparameters
- Data loading settings
- Checkpoint management
- File paths and metadata handling
"""

import pathlib
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


class DataConfig(BaseModel, frozen=True):
    """Configuration for data loading and preprocessing.

    Controls batch sizes, sequence lengths, and data processing parameters
    for both audio and text modalities.

    Attributes:
        text_length: Maximum length of text sequences, must be multiple of 128
        audio_length: Maximum length of audio sequences, must be multiple of 128
        channels: Number of audio channels
        text_pad_value: Value used for padding text sequences
        audio_eos_value: Value used for end of audio sequences
        audio_bos_value: Value used for beginning of audio sequences
        audio_pad_value: Value used for padding audio sequences
        delay_pattern: List of delay values for each channel
    """

    text_length: Annotated[int, BeforeValidator(lambda x: (x + 127) // 128 * 128)] = Field(gt=0, multiple_of=128)
    audio_length: Annotated[int, BeforeValidator(lambda x: (x + 127) // 128 * 128)] = Field(gt=0, multiple_of=128)
    channels: int = Field(default=9, gt=0, multiple_of=1)
    text_pad_value: int = Field(default=0)
    audio_eos_value: int = Field(default=1024)
    audio_pad_value: int = Field(default=1025)
    audio_bos_value: int = Field(default=1026)
    delay_pattern: list[Annotated[int, Field(ge=0)]] = Field(default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15])

    def __hash__(self) -> int:
        """Generate a hash based on all fields of the config.

        Since the class is frozen, we can safely hash all fields.
        For the delay_pattern list, we convert it to a tuple since lists are not hashable.
        """
        return hash(
            (
                self.text_length,
                self.audio_length,
                self.channels,
                self.text_pad_value,
                self.audio_pad_value,
                self.audio_bos_value,
                self.audio_eos_value,
                tuple(self.delay_pattern),  # Convert list to tuple for hashing
            )
        )


class EncoderConfig(BaseModel, frozen=True):
    """Configuration for the encoder component of the Nari model.

    Defines the architecture and dimensions of the encoder, including
    attention mechanisms and layer specifications.

    Attributes:
        n_layer: Number of transformer layers
        n_embd: Embedding dimension, must be divisible by n_head
        n_hidden: Hidden dimension, must be divisible by n_embd
        n_head: Number of attention heads
        head_dim: Dimension per attention head (computed automatically)
        mlp_activations: List of activation functions for the MLP
        use_pre_norm: Whether to use pre-normalization
    """

    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    n_head: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)


class DecoderConfig(BaseModel, frozen=True):
    """Configuration for the decoder component of the Nari model.

    Defines the architecture and dimensions of the decoder, including
    grouped-query attention (GQA) and cross-attention mechanisms.

    Attributes:
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        n_hidden: Hidden dimension
        gqa_query_heads: Number of query heads for grouped-query attention
        cross_query_heads: Number of query heads for cross-attention
        kv_heads: Number of key-value heads (must divide gqa_query_heads)
        gqa_head_dim: Dimension per GQA head (computed automatically)
        cross_head_dim: Dimension per cross-attention head (computed automatically)
    """

    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    gqa_query_heads: int = Field(gt=0)
    cross_query_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    gqa_head_dim: int = Field(gt=0)
    cross_head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)


class ModelConfig(BaseModel, frozen=True):
    """Main configuration container for the Nari model architecture.

    Combines encoder and decoder configurations with vocabulary sizes
    and dropout settings.

    Attributes:
        encoder: Configuration for the encoder component
        decoder: Configuration for the decoder component
        src_vocab_size: Size of the source vocabulary
        tgt_vocab_size: Size of the target vocabulary
        dropout: Dropout probability during training
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    src_vocab_size: int = Field(default=128, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    weight_dtype: str = Field(default="float32", description="Weight precision")
    rope_min_timescale: int = Field(default=1, description="Timesclae For global Attention")
    rope_max_timescale: int = Field(default=10_000, description="Timesclae For global Attention")


class TrainingConfig(BaseModel, frozen=True):
    """Training process configuration and hyperparameters.

    Controls learning rate scheduling, optimization parameters,
    and training behavior.

    Attributes:
        batch_size: Number of samples per batch, must be positive and multiple of 8
        max_steps: Maximum number of training steps
        eval_every_steps: Number of steps between evaluations
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps for learning rate
        lr_decay_steps: Steps after which learning rate decay begins
        min_lr: Minimum learning rate after decay
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        unconditional_training_prob: Probability of unconditional training
        g_accum_iters: Number of gradient accumulation iterations
    """

    dtype: str = Field(default="bfloat16", description="Activation precision")
    logits_dot_in_fp32: bool = Field(default=False)


class NariConfig(BaseModel, frozen=True):
    """Master configuration for the Nari model.

    Combines all sub-configurations into a single validated configuration
    object, including model architecture, training parameters, and paths.

    Attributes:
        version: Configuration version string
        created_at: Creation timestamp
        random_seed: Random seed for reproducibility
        model: Model architecture configuration
        training: Training process configuration
        data_paths: File system paths configuration
        data: Data loading configuration
    """

    version: str = Field(default="1.0")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig


def save_config(path: pathlib.Path, config: NariConfig) -> None:
    """Save configuration to disk.

    Args:
        path: Path to the configuration file
        config: Complete Nari configuration to save
    """
    assert path.is_file()
    assert path.suffix == ".json"
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = config.model_dump_json(indent=2)
    with open(path, "w") as f:
        f.write(config_json)


def load_config(path: pathlib.Path) -> NariConfig | None:
    """Load configuration from disk.

    Deserializes a configuration from JSON and validates it.

    Args:
        path: Path to the configuration file

    Returns:
        Loaded and validated configuration, or None if file not found
    """
    try:
        assert path.is_file()
        assert path.suffix == ".json"
        with open(path, "r") as f:
            content = f.read()
        return NariConfig.model_validate_json(content)
    except FileNotFoundError:
        return None
