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
import secrets
from datetime import datetime
from typing import Annotated, Any

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


class EmbedConfig(BaseModel, frozen=True):
    """Configuration for the embedding (and related) component of the Nari model.

    Defines the architecture and dimensions of the embedding.
    """

    use_iota_embed: bool = Field(
        default=False,
        description="Use iota operator in Embed",
    )
    normalize_embedding_logits: bool = Field(
        default=True,
        description="Whether to normalize pre-softmax logits if logits_via_embedding is true",
    )


class LinearConfig(BaseModel, frozen=True):
    """Configuration for the linear component of the Nari model.

    Defines the architecture and dimensions of the linear component.
    """

    fused_mlp: bool = Field(default=True)
    mlp_activations: list[str] = Field(default=["silu", "linear"])


class ExpertConfig(BaseModel, frozen=True):
    """Configuration for the expert component of the Nari model.

    Defines the architecture and dimensions of the expert component.
    """

    num_experts: int = Field(default=1, gt=0)
    num_experts_per_tok: int = Field(default=1, gt=0)
    moe_mlp_dim: int = Field(default=2048, gt=0)
    shared_experts: int = Field(default=1, gt=0)


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
    linear: LinearConfig
    embed: EmbedConfig
    expert: ExpertConfig
    src_vocab_size: int = Field(default=128, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    scan_layers: bool = Field(
        default=True,
        description="Recommended to set to false when using pipeline parallelism, instead scanning the PP iterations.",
    )
    param_scan_axis: int = Field(
        default=1,
    )
    weight_dtype: str = Field(default="float32", description="Weight precision")
    attention: str = Field(
        default="flash",
        description="Supported attention: autoselected, dot_product, flash, cudnn_flash_te, paged",
    )
    attention_type: str = Field(
        default="global",
        description="Supported attention_type: global, local_sliding, mla",
    )
    sparse_matmul: bool = Field(default=True)
    norm_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    sa_block_q: int = Field(default=512)
    sa_block_kv: int = Field(default=512)
    sa_block_kv_compute: int = Field(default=512)
    sa_block_q_dkv: int = Field(default=512)
    sa_block_kv_dkv: int = Field(default=512)
    sa_block_kv_dkv_compute: int = Field(default=512)
    sa_block_q_dq: int = Field(default=512)
    sa_block_kv_dq: int = Field(default=512)
    sa_use_fused_bwd_kernel: bool = Field(default=False)
    sa_q_layout: str = Field(default="HEAD_DIM_MINOR")
    sa_k_layout: str = Field(default="HEAD_DIM_MINOR")
    sa_v_layout: str = Field(default="HEAD_DIM_MINOR")
    use_chunked_prefill: bool = Field(default=False)
    pagedattn_num_pages: int = Field(default=64)
    pagedattn_tokens_per_page: int = Field(default=32)
    pagedattn_pages_per_compute_block: int = Field(default=8)
    prefill_chunk_size: int = Field(default=256)
    fused_qkv: bool = Field(default=False)
    model_name: str = Field(
        default="default",
        description="Model name, used to select the correct RoPE parameters",
    )
    # RoPE parameters
    rope_type: str = Field(default="default", description="one of 'default', 'llama3.1' or 'yarn'")
    rope_min_timescale: int = Field(default=1, description="Timesclae For global Attention")
    rope_max_timescale: int = Field(default=10_000, description="Timesclae For global Attention")
    local_rope_max_timescale: int = Field(
        default=-1,
        description="If positive used for local window Attention, otherwise `rope_max_timescale` is used for both local and global",
    )
    use_untrainable_positional_embedding: bool = Field(
        default=False,
    )
    # Yarn parameters
    max_position_embeddings: int = Field(default=163840)
    original_max_position_embeddings: int = Field(default=4096)
    rope_factor: int = Field(default=40)
    beta_fast: float = Field(default=32)
    beta_slow: float = Field(default=1)
    mscale: float = Field(default=1.0)

    # KV Cache
    prefill_cache_axis_order: str = Field(
        default="1,2,0,3",
        description="Logical layout: 0,1,2,3 ; CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV",
    )
    ar_cache_axis_order: str = Field(
        default="1,2,0,3",
        description="Logical layout: 0,1,2,3 ; CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV",
    )
    compute_axis_order: str = Field(
        default="0,1,2,3",
        description="Default layout: 0,1,2,3 ; BATCH, LENGTH, HEAD, D_KV",
    )


class QuantizationConfig(BaseModel, frozen=True):
    """Configuration for quantization.

    Defines the quantization configuration for the model.
    """

    quantization: str = Field(
        default="",
        description="Quantization configuration. Valid values: 'int8', 'intmp', 'fp8', 'nanoo_fp8', 'aqt_fp8'. Defaults to null implying bf16.",
    )
    quant_cfg_path: str = Field(default="", description="Path to file with quantization config for intmp.")
    replicate_quant_scale: bool = Field(
        default=False,
        description="Used to replicate the quantization scale to avoid the inefficient XLA fusion for 2d sharding.",
    )
    quantize_kvcache: bool = Field(default=False, description="Set to True to quantize KV Cache values")
    kv_quant_axis: str = Field(
        default="heads_and_dkv",
        description="Quantization axis for KV cache. Valid values: '' (when quantize_kvcache is False), "
        "'dkv' (quantize over cache_kv dimension, better accuracy but slower), "
        "'heads_and_dkv' (quantize over cache_heads and cache_kv axes, faster computation)",
    )
    kv_quant_dtype: str = Field(default="int8", description="Quantization data type for KV cache.")
    quantization_local_shard_count: int = Field(
        default=-1,
        description="Shard the range finding operation for quantization. By default this is set to number of slices",
    )


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

    global_batch_size: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    enable_checkpointing: bool = Field(default=True)
    eval_every_steps: int = Field(gt=0)
    val_steps: int = Field(gt=0)
    val_d_steps: int = Field(gt=0)
    learning_rate: float = Field(ge=0.0)
    warmup_steps: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)
    lr_decay_steps: int = Field(gt=0)
    min_lr: float = Field(ge=0.0)
    beta1: float = Field(ge=0.0, lt=1.0)
    beta2: float = Field(ge=0.0, lt=1.0)
    unconditional_training_prob: float = Field(ge=0.0, lt=1.0)
    g_accum_iters: int = Field(gt=0)
    loss_scale: float = Field(default=1.0)
    dtype: str = Field(default="bfloat16", description="Activation precision")
    activations_in_float32: bool = Field(
        default=False,
        description="Sets activations to float32 before nonlinearity it true, else dtype",
    )
    matmul_precision: str = Field(default="default")
    optimizer_memory_host_offload: bool = Field(
        default=False,
        description="Offload optimizer memory to host if true, else device",
    )
    logits_dot_in_fp32: bool = Field(default=False)
    cast_logits_to_fp32: bool = Field(default=True)
    encoder_remat_policy: str = Field(default="full")
    decoder_remat_policy: str = Field(default="full")
    sharding_tolerance: float = Field(default=0.02)


class ParallelismConfig(BaseModel, frozen=True):
    """Configuration for parallelism strategies.

    Defines how the model should be distributed across multiple devices:
    - REPLICATE: Full model copy on each device
    - SHARD: Model partitioned across devices
    """

    data_parallelism: int = Field(default=8)  # data is split across 8 devices
    fsdp_parallelism: int = Field(default=-1)  # recommended ICI axis to be auto-sharded
    fsdp_transpose_parallelism: int = Field(default=1)
    sequence_parallelism: int = Field(default=1)
    context_parallelism: int = Field(default=1)
    tensor_parallelism: int = Field(default=1)
    tensor_transpose_parallelism: int = Field(default=1)
    tensor_sequence_parallelism: int = Field(default=1)
    autoregressive_parallelism: int = Field(default=1)
    pipeline_parallelism: int = Field(default=1)
    expert_parallelism: int = Field(default=1)
    mesh_axes: list[str] = Field(default=[])
    data_sharding: list[list[str]] = Field(default=[[]])
    logical_axis_rules: list[list[Any]] = Field(default=[[]])
    # Pipeline Parallelism Only
    num_layers_per_pipeline_stage: int = Field(
        default=1,
        description="The number of layers per pipeline stage. If set to 1, the pipeline parallelism is disabled.",
    )
    num_pipeline_repeats: int = Field(
        default=-1,
        description="The number of pipeline repeats. If set to -1, the number of repeats will be set to num_decoder_layers / (num_pipeline_stages * num_layers_per_pipeline_stage)",
    )
    num_pipeline_microbatches: int = Field(
        default=-1,
        description="The number of pipeline microbatches. If set to -1, the number of microbatches will be set to the number of pipeline stages.",
    )
    pipeline_delay_activation_forwarding: bool = Field(
        default=False,
        description="This delays the activation forwarding one loop iteration simplifying XLA's task of overlapping since the communication and compute in each iteration are now independent. However this comes at the cost of doubling the pipeline bubble, and you must set the number of microbatches to at least 2 * num_stages (the minimum 2 * num_stages is set by default with this delay).",
    )
    pipeline_fsdp_ag_once: bool = Field(
        default=False,
        description="If set to true then all gather all of the weights over FSDP before the first pipeline iteration.",
    )
    scan_pipeline_iterations: bool = Field(default=True)
    set_remat_policy_on_pipeline_iterations: bool = Field(default=True)
    set_remat_policy_on_layers_per_stage: bool = Field(default=False)


class InferenceConfig(BaseModel, frozen=True):
    """Configuration for inference.

    Defines the configuration for inference.
    """

    strategy: str = Field(default="greedy")
    nucleus_p: float = Field(default=-1)
    top_k: int = Field(default=0)
    temperature: float = Field(default=1.0)
    guidance_scale: float = Field(default=3.0)


class ProfilerConfig(BaseModel, frozen=True):
    """Configuration for the profiler.

    Defines the configuration for the profiler.
    """

    enabled: bool = Field(default=False)
    period: int = Field(default=100)
    skip_first_n_steps: int = Field(default=0)
    steps: int = Field(default=300)


class DataPaths(BaseModel, frozen=True):
    """Configuration for dataset and checkpoint file paths.

    Manages file system paths for training data, validation data,
    and model checkpoints.

    Attributes:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        val_d_dir: Directory containing validation data
    """

    train_dir: pathlib.Path
    val_dir: pathlib.Path
    val_d_dir: pathlib.Path


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
    random_seed: int = Field(default_factory=lambda: secrets.randbits(32))
    model: ModelConfig
    training: TrainingConfig
    data_paths: DataPaths
    data: DataConfig
    parallelism: ParallelismConfig
    quantization: QuantizationConfig
    inference: InferenceConfig
    profiler: ProfilerConfig


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
