from dataclasses import dataclass

import torch

from .config import DiaConfig


def create_attn_mask(
    q_padding_mask_1d: torch.Tensor,
    k_padding_mask_1d: torch.Tensor,
    device: torch.device,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Creates the attention mask (self or cross) mimicking JAX segment ID logic.
    """
    B1, Tq = q_padding_mask_1d.shape
    B2, Tk = k_padding_mask_1d.shape
    assert B1 == B2, "Query and key batch dimensions must match"

    p_mask_q = q_padding_mask_1d.unsqueeze(2)  # Shape [B, Tq, 1]
    p_mask_k = k_padding_mask_1d.unsqueeze(1)  # Shape [B, 1, Tk]

    # Condition A: Non-padding query attends to non-padding key
    non_pad_attends_non_pad = p_mask_q & p_mask_k  # Shape [B, Tq, Tk]

    # Condition B: Padding query attends to padding key
    pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

    # Combine: True if padding status is compatible (both non-pad OR both pad)
    mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

    if is_causal:
        assert Tq == Tk, "Causal mask requires query and key sequence lengths to be equal"
        causal_mask_2d = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))  # Shape [Tq, Tk]
        causal_mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]
        return causal_mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]
    else:
        return mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]


@dataclass
class EncoderInferenceState:
    """Parameters specifically for encoder inference."""

    max_seq_len: int
    device: torch.device
    dtype: torch.dtype
    positions: torch.Tensor
    padding_mask: torch.Tensor
    attn_mask: torch.Tensor

    @classmethod
    def new(cls, config: DiaConfig, cond_src: torch.Tensor) -> "EncoderInferenceState":
        """Creates EtorchrInferenceParams from DiaConfig and a device."""
        device = cond_src.device

        positions = torch.arange(config.data.text_length, device=device).to(torch.long).unsqueeze(0).expand(2, -1)
        padding_mask = (cond_src != config.data.text_pad_value).to(device).expand(2, -1)
        attn_mask = create_attn_mask(padding_mask, padding_mask, device, is_causal=False)

        return cls(
            max_seq_len=config.data.text_length,
            device=device,
            dtype=config.training.dtype,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )


class KVCache:
    def __init__(self, num_heads: int, max_len: int, head_dim: int, device: torch.device):
        self.k = torch.zeros((2, num_heads, max_len, head_dim), device=device)
        self.v = torch.zeros((2, num_heads, max_len, head_dim), device=device)
        self.current_idx = 0
        self.max_len = max_len

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.current_idx < self.max_len
        next_idx = self.current_idx + k.shape[2]
        assert next_idx < self.max_len
        self.k[:, :, self.current_idx : next_idx, :] = k
        self.v[:, :, self.current_idx : next_idx, :] = v
        self.current_idx = next_idx
        return self.k[:, :, :next_idx, :], self.v[:, :, :next_idx, :]


@dataclass
class DecoderInferenceState:
    """Parameters specifically for decoder inference."""

    max_seq_len: int
    device: torch.device
    dtype: torch.dtype
    enc_positions: torch.Tensor
    enc_out: torch.Tensor
    dec_positions: torch.Tensor
    dec_cross_attn_mask: torch.Tensor
    self_attn_cache: list[KVCache]
    cross_attn_cache: list[KVCache]
    step: int
    generated_tokens: torch.Tensor

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def new(
        cls, config: DiaConfig, enc_state: EncoderInferenceState, enc_out: torch.Tensor
    ) -> "DecoderInferenceState":
        """Creates DecoderInferenceParams from DiaConfig and a device."""
        device = enc_out.device
        max_audio_len = config.data.audio_length

        step = 0
        dec_positions = torch.full((2, 1), fill_value=step, dtype=torch.long, device=device)
        tgt_padding_mask = torch.ones((2, 1), dtype=torch.bool, device=device)
        dec_cross_attn_mask = create_attn_mask(tgt_padding_mask, enc_state.padding_mask, device, is_causal=False)
        generated_tokens = torch.full(
            (1, max_audio_len, config.data.channels),
            fill_value=config.data.audio_pad_value,
            dtype=torch.long,
            device=device,
        )

        self_attn_cache = [
            KVCache(
                config.model.decoder.kv_heads,
                max_audio_len,
                config.model.decoder.gqa_head_dim,
                device,
            )
            for _ in range(config.model.decoder.n_layer)
        ]

        cross_attn_cache = [
            KVCache(
                config.model.decoder.cross_query_heads,
                config.data.text_length,
                config.model.decoder.cross_head_dim,
                device,
            )
            for _ in range(config.model.decoder.n_layer)
        ]

        return cls(
            max_seq_len=max_audio_len,
            device=device,
            dtype=config.training.dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            dec_cross_attn_mask=dec_cross_attn_mask,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=cross_attn_cache,
            step=step,
            generated_tokens=generated_tokens,
        )

    def update_step(self, dec_out: torch.Tensor, d_step: int | None = None, apply_mask: bool = False):
        """Mutates the state for the next decoding step."""
        step_to = self.step + dec_out.shape[1]
        if apply_mask:
            mask = self.generated_tokens[:, self.step : step_to, :] == self.config.data.audio_pad_value
            self.generated_tokens[:, self.step : step_to, :] = torch.where(
                mask, dec_out, self.generated_tokens[:, self.step : step_to, :]
            )
        else:
            self.generated_tokens[:, self.step : step_to, :] = dec_out

        self.step += dec_out.shape[1] if d_step is None else d_step
        self.dec_positions = torch.full((2, 1), fill_value=self.step, dtype=torch.long, device=self.device)
