# Inference script for Nari (text in -> audio out)
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import dac
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .audio import audio_to_codebook, codebook_to_audio
from .config import NariConfig, load_config
from .model import KVCache, Nari


def load_model_and_config(config_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[Nari, NariConfig]:
    """Loads the Nari model and its configuration."""
    config = load_config(config_path)
    model = Nari(config)
    print(f"Instantiated Nari model with config from {config_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_result = model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded model weights from {checkpoint_path}. Result: {load_result}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")

    model.to(device)
    model.eval()
    return model, config


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
    # This implementation follows Jax TPU splash attention kernel
    mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

    if is_causal:
        # Ensure causality for self-attention (Tq == Tk)
        assert Tq == Tk, "Causal mask requires query and key sequence lengths to be equal"
        # Standard lower-triangular causal mask (True means allow)
        causal_mask_2d = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))  # Shape [Tq, Tk]
        causal_mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]
        return causal_mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk] for broadcasting across heads
    else:
        # For cross-attention or non-causal self-attention
        return mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk] for broadcasting across heads


def _prepare_text_input(
    txt_file_path: str, config: NariConfig, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encodes text prompt, pads, and creates attention mask and positions."""
    text_pad_value = config.data.text_pad_value
    max_len = config.data.text_length

    # Turn [S1] and [S2] into special tokens using standard Python
    with open(txt_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    byte_text = raw_text.encode("utf-8")
    replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
    text_tokens = list(replaced_bytes)

    current_len = len(text_tokens)
    padding_needed = max_len - current_len
    if padding_needed <= 0:
        text_tokens = text_tokens[:max_len]
        padded_text_np = np.array(text_tokens, dtype=np.uint8)
    else:
        padded_text_np = np.pad(
            text_tokens,
            (0, padding_needed),
            mode="constant",
            constant_values=text_pad_value,
        ).astype(np.uint8)

    src_tokens = torch.from_numpy(padded_text_np).to(torch.long).to(device).unsqueeze(0)  # [1, S]
    src_positions = torch.arange(max_len, device=device).to(torch.long).unsqueeze(0)  # [1, S]

    src_padding_mask = (src_tokens != text_pad_value).to(device)  # [1, S]

    enc_self_attn_mask = create_attn_mask(src_padding_mask, src_padding_mask, device, is_causal=False)  # [1, S, S]

    return src_tokens, src_positions, src_padding_mask, enc_self_attn_mask


def sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    cfg_filter_top_k: Optional[int] = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if use_cfg_filter and cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -float("inf"))

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        # Calculate indices to remove based on top_p
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        # Shift the mask to the right to keep the first token above the threshold
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0  # Always keep the most probable token

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -float("inf"))

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


@torch.inference_mode()
def generate(
    model: Nari,
    config: NariConfig,
    txt_file_path: str,
    max_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    device: torch.device,
    use_torch_compile: bool,
    cfg_filter_top_k: int,
    audio_prompt_path: Optional[str] = None,
    dac_model: Optional[dac.DAC] = None,
) -> torch.Tensor:
    """
    Generates audio from a text prompt (and optional audio prompt) using the Nari model.

    Returns:
        A tensor of generated audio codes (shape: [max_tokens, num_channels]).
    """
    num_channels = config.data.channels
    audio_bos_value = config.data.audio_bos_value
    audio_eos_value = config.data.audio_eos_value
    audio_pad_value = config.data.audio_pad_value
    delay_pattern = config.data.delay_pattern
    delay_tensor = torch.tensor(delay_pattern, dtype=torch.long, device=device)
    model.eval()

    # 1. Prepare Encoder Input
    (
        cond_src_BxS,
        cond_src_positions_BxS,
        cond_src_padding_mask_BxS,
        cond_enc_self_attn_mask_Bx1xSxS,
    ) = _prepare_text_input(txt_file_path, config, device)

    unc_src_BxS = torch.zeros_like(cond_src_BxS)
    src_BxS = torch.cat([unc_src_BxS, cond_src_BxS], dim=0)
    src_positions_BxS = cond_src_positions_BxS.expand(2, -1)
    src_padding_mask_BxS = cond_src_padding_mask_BxS.expand(2, -1)
    enc_self_attn_mask_Bx1xSxS = cond_enc_self_attn_mask_Bx1xSxS.expand(2, -1, -1, -1)

    # 2. Encoder Pass
    # with torch.autocast(device_type="cuda", dtype=forward_dtype):
    encoder_out = model.encoder(
        x_ids=src_BxS,
        src_positions=src_positions_BxS,
        deterministic=True,
        attn_mask=enc_self_attn_mask_Bx1xSxS,
    )  # Shape: (B, S, E)

    print(f"Encoder Output Shape: {encoder_out.shape}")

    # 3. Prepare Decoder Inputs
    # 3-1. Allocate KV Cache (Static)
    decoder_cross_attention_cache: List[KVCache] = model.decoder.precompute_cross_attention_kv(
        max_tokens, encoder_out, src_positions_BxS
    )

    decoder_self_attention_cache: List[KVCache] = []
    for _ in range(model.decoder.num_layers):
        decoder_self_attention_cache.append(
            KVCache(
                16,
                max_tokens,
                128,
                device,
            )
        )

    # 3-2. Initialize Decoder Inputs
    generated_BxTxC = torch.full(
        (2, 1, num_channels),
        fill_value=audio_bos_value,
        dtype=torch.long,
        device=device,
    )

    current_step = 0
    prompt_len_inc_bos = 1  # Start with BOS length

    # 3-3. Load Audio Prompt (if provided)
    if audio_prompt_path is not None:
        audio_prompt, sr = torchaudio.load(audio_prompt_path, channels_first=True)  # C, T
        if sr != 44100:  # Resample to 44.1kHz
            audio_prompt = torchaudio.functional.resample(audio_prompt, sr, 44100)
        audio_prompt = audio_prompt.to(device).unsqueeze(0)  # 1, C, T
        audio_prompt = audio_to_codebook(dac_model, audio_prompt, data_config=config.data)
        generated_BxTxC = torch.cat([generated_BxTxC, audio_prompt.expand(2, -1, -1)], dim=1)

        print("\n--- Running Decoder Prefill Step ---")
        prefill_len = generated_BxTxC.shape[1]
        prompt_len_inc_bos = prefill_len
        prefill_tgt_pos = torch.arange(prefill_len, device=device).unsqueeze(0).expand(2, -1)
        prefill_tgt_padding_mask = (generated_BxTxC != audio_pad_value).any(dim=2)

        prefill_self_attn_mask = create_attn_mask(
            prefill_tgt_padding_mask,
            prefill_tgt_padding_mask,
            device,
            is_causal=True,
        )
        prefill_cross_attn_mask = create_attn_mask(
            prefill_tgt_padding_mask,
            src_padding_mask_BxS,
            device,
            is_causal=False,
        )

        _ = model.decoder.forward(
            tgt_ids_BxTxC=generated_BxTxC,
            encoder_out=encoder_out,
            tgt_positions=prefill_tgt_pos,
            src_positions=src_positions_BxS,
            deterministic=True,
            self_attn_mask=prefill_self_attn_mask,
            cross_attn_mask=prefill_cross_attn_mask,
            self_attention_cache=decoder_self_attention_cache,
            cross_attention_cache=decoder_cross_attention_cache,
        )

        current_step = prefill_len - 1
        print(f"Prefill finished. Starting AR generation from step {current_step + 1}.")

    # 4. Autoregressive Generation Loop
    eos_detected_channel_0 = False
    eos_countdown = -1
    extra_steps_after_eos = 30
    # Make generated_BxTxC a fixed size tensor
    # Length is either 1 + max tokens or 1 + prompt len + max tokens
    generated_BxTxC = torch.cat(
        [
            generated_BxTxC,
            torch.full(
                (2, max_tokens, num_channels),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )

    decode_step = model.decoder.decode_step
    if use_torch_compile:
        decode_step = torch.compile(
            model.decoder.decode_step,
            mode="default",
        )

    tgt_padding_mask = (generated_BxTxC[:, -1, :].unsqueeze(1) != audio_pad_value).any(dim=2).to(device)  # [B, 1]
    # Generated tokens are never PAD, so we use fixed mask
    decoder_cross_attn_mask = create_attn_mask(
        tgt_padding_mask,  # Query mask [B, 1]
        src_padding_mask_BxS,  # Key mask [B, S]
        device,
        is_causal=False,
    )  # [B, 1, 1, S]

    step = 0
    start_gen_time = start_loop_time = time.time()
    for step in range(current_step, current_step + max_tokens):
        tgt_ids_Bx1xC = generated_BxTxC[:, step, :].unsqueeze(1)
        tgt_pos_Bx1 = torch.full(
            (2, 1),
            fill_value=step,
            dtype=torch.long,
            device=device,
        )

        # print(f"Step {step} | Decoder input: {tgt_ids_Bx1xC}")

        logits_Bx1xCxV, new_cache = decode_step(
            tgt_ids_Bx1xC=tgt_ids_Bx1xC,
            tgt_pos_Bx1=tgt_pos_Bx1,
            encoder_out=encoder_out,
            self_attn_mask=None,
            cross_attn_mask=decoder_cross_attn_mask,
            self_attention_cache=decoder_self_attention_cache,
            cross_attention_cache=decoder_cross_attention_cache,
        )

        for i, layer_cache in enumerate(decoder_self_attention_cache):
            layer_cache.update_cache(new_cache[i][0], new_cache[i][1])

        V = config.model.tgt_vocab_size
        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]  # B, C, V
        uncond_logits_CxV = logits_last_BxCxV[0, :, :]
        cond_logits_CxV = logits_last_BxCxV[1, :, :]

        cfg_logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)

        logits_CxV = cfg_logits_CxV.reshape((-1, V))  # C, V
        logits_CxV[:, 1025:] = -float("inf")

        # Sample next token
        pred_C = sample_next_token(
            logits_CxV.float(),
            temperature=temperature,
            top_p=top_p,
            use_cfg_filter=use_cfg_filter,
            cfg_filter_top_k=cfg_filter_top_k,
        )

        generation_step_index = step - current_step
        if audio_prompt_path is None:
            pred_C = torch.where(
                generation_step_index >= delay_tensor,
                pred_C,
                audio_bos_value,
            )

        # print(f"Step {step} | Decoder output: {pred_C}")

        generated_BxTxC[:, step + 1, :] = pred_C.unsqueeze(0).expand(2, -1)

        if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
            print(f"EOS detected on channel 0 at step {step}.")
            eos_detected_channel_0 = True
            eos_countdown = extra_steps_after_eos

        if eos_countdown > 0:
            eos_countdown -= 1
            if eos_countdown == 0:
                print("Finished extra steps after EOS. Stopping generation.")
                break
        elif eos_detected_channel_0 and step >= current_step + max_tokens - 1:
            print("Max tokens reached after EOS detected.")

        # Report progress based on tokens generated *in this run*
        generation_step_index = step - current_step + 1
        total_steps_to_generate = max_tokens
        if generation_step_index % 50 == 0 or generation_step_index == total_steps_to_generate:
            elapsed_time = time.time() - start_loop_time
            steps_per_sec = 50 / elapsed_time if elapsed_time > 0 else 0
            print(f"Generated step {generation_step_index}/{total_steps_to_generate} ({steps_per_sec:.2f} steps/s)")
            start_loop_time = time.time()

    avg_steps_per_sec = (step - current_step + 1) / (time.time() - start_gen_time)
    print(
        f"\nGeneration loop finished in {time.time() - start_gen_time:.2f}s at step {step}. "
        f"({avg_steps_per_sec:.2f} steps/s)\n"
    )

    output_codes = generated_BxTxC[:, prompt_len_inc_bos : step + 1, :]

    return output_codes[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--checkpoint", type=str, default="./weights.pth")
    parser.add_argument("--txt_file", type=str, default="./example.txt")
    parser.add_argument("--output_path", type=str, default="./example_compiled.mp3")
    parser.add_argument("--max_tokens", type=int, default=2600)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--use_cfg_filter", type=bool, default=True)
    parser.add_argument("--cfg_filter_top_k", type=int, default=50)
    parser.add_argument("--use_torch_compile", type=bool, default=False)
    parser.add_argument(
        "--audio_prompt",
        type=str,
        default=None,
        help="Path to audio prompt file or None",
    )
    # parser.add_argument("--audio_prompt_file", type=str | None, default=None)
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)
    CHECKPOINT_PATH = Path(args.checkpoint)
    txt_file_path = args.txt_file
    audio_output_path = args.output_path
    max_tokens = args.max_tokens
    cfg_scale = args.cfg_scale
    temperature = args.temperature
    top_p = args.top_p
    use_cfg_filter = args.use_cfg_filter
    cfg_filter_top_k = args.cfg_filter_top_k
    audio_prompt_path = args.audio_prompt
    use_torch_compile = args.use_torch_compile
    # --- Setup ---
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        exit(1)
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit(1)
    if not Path(txt_file_path).exists():
        print(f"Error: Text input file not found at {txt_file_path}")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    try:
        model, config = load_model_and_config(CONFIG_PATH, CHECKPOINT_PATH, device)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error loading model: {e}")
        exit(1)

    dac_model = None
    if audio_prompt_path is not None:
        print("Loading DAC model...")
        model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(model_path).to(device)

    print("\n--- Starting Generation ---")
    try:
        generated_codes = generate(
            model=model,
            config=config,
            txt_file_path=txt_file_path,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            use_cfg_filter=use_cfg_filter,
            cfg_filter_top_k=cfg_filter_top_k,
            device=device,
            audio_prompt_path=audio_prompt_path,
            dac_model=dac_model,
            use_torch_compile=use_torch_compile,
        )

        if generated_codes.numel() > 0:
            if dac_model is None:
                print("Loading DAC model...")
                model_path = dac.utils.download(model_type="44khz")
                dac_model = dac.DAC.load(model_path).to(device)
            audio = codebook_to_audio(
                generated_codes.transpose(1, 0),
                dac_model,
                config.data.delay_pattern,
                B=1,
                T=max_tokens,
                C=config.data.channels,
            )
            sf.write(audio_output_path, audio.cpu().numpy().squeeze(), 44100)
        else:
            print("\nGeneration finished, but no tokens were produced.")

    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")
        import traceback

        traceback.print_exc()
