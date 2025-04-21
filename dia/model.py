import dac
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download

from .audio import (
    apply_audio_delay,
    build_delay_indices,
    build_revert_indices,
    revert_audio_delay,
)
from .config import DiaConfig
from .layers import Decoder, Encoder
from .state import DecoderInferenceState, EncoderInferenceState


SPEAKER_TOKENS = {
    b"[S1]": b"\x01",
    b"[S2]": b"\x02",
}

# --- Helper Functions (moved from inference.py) ---


def sample_next_token(
    logits_BCxV: torch.Tensor,  # Input shape [B*C, V] or [C, V] if B=1
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:  # Output shape [B*C] or [C] if B=1
    """Samples the next token index based on logits using temperature and top-p sampling.

    Optionally applies top-k filtering before sampling if `cfg_filter_top_k` is set.
    Handles the case where temperature is 0 (argmax).

    Args:
        logits_BCxV: The input logits tensor. Expected shape is [B*C, V] or [C, V]
                     if batch size B is 1. V is the vocabulary size.
        temperature: The temperature for sampling. Higher values increase randomness,
                     0.0 corresponds to argmax.
        top_p: The cumulative probability threshold for nucleus sampling. If < 1.0,
               only the most probable tokens with cumulative probability <= top_p
               are considered.
        cfg_filter_top_k: If not None, filters the logits to only the top k most
                          probable tokens *before* applying temperature and top-p.

    Returns:
        A tensor containing the sampled token indices. Shape is [B*C] or [C] if B=1.
    """
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    if cfg_filter_top_k is not None:
        _, top_k_indices = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -float("inf"))

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove, -float("inf"))

    final_probs = torch.softmax(logits_BCxV, dim=-1)
    sampled_indices = torch.multinomial(final_probs, num_samples=1)

    return sampled_indices.squeeze(-1)


class Dia(nn.Module):
    """PyTorch Dia Model for Text-to-Multi-Channel Audio Synthesis.

    This model combines an Encoder-Decoder Transformer architecture with a
    pre-trained DAC audio codec to generate multi-channel audio based on
    input text. It handles text encoding, speaker tokens, audio channel delay
    patterns, and autoregressive generation with classifier-free guidance.

    Attributes:
        config (DiaConfig): The configuration object containing hyperparameters.
        device (torch.device): The device the model is loaded on.
        encoder (Encoder): The text encoder module.
        decoder (Decoder): The audio token decoder module.
        dac_model (dac.DAC): The pre-trained DAC audio codec model.
        _apply_delay_t_idx (torch.Tensor): Precomputed time indices for applying delay.
        _apply_delay_indices (torch.Tensor): Precomputed gather indices for applying delay.
        _revert_delay_t_idx (torch.Tensor): Precomputed time indices for reverting delay.
        _revert_delay_indices (torch.Tensor): Precomputed gather indices for reverting delay.
    """

    def __init__(self, config: DiaConfig, device: torch.device = torch.device("cuda")):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            device: The device to load the model onto.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        apply_t_idx, apply_indices = build_delay_indices(
            B=1, T=config.data.audio_length, C=config.data.channels, delay_pattern=config.data.delay_pattern
        )
        revert_t_idx, revert_indices = build_revert_indices(
            B=1, T=config.data.audio_length, C=config.data.channels, delay_pattern=config.data.delay_pattern
        )

        self.register_buffer("_apply_delay_t_idx", apply_t_idx, persistent=False)
        self.register_buffer("_apply_delay_indices", apply_indices, persistent=False)
        self.register_buffer("_revert_delay_t_idx", revert_t_idx, persistent=False)
        self.register_buffer("_revert_delay_indices", revert_indices, persistent=False)

        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(device)
            dac_model.eval()
        except Exception as e:
            raise RuntimeError("Error loading DAC model") from e
        self.dac_model = dac_model

    @classmethod
    def from_local(cls, config_path: str, checkpoint_path: str, device: torch.device = torch.device("cuda")) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            device: The device to load the model onto.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        model = cls(config, device)

        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        model.to(device)
        model.eval()
        return model

    @classmethod
    def from_pretrained(
        cls, model_name: str = "nari-labs/Dia-1.6B", device: torch.device = torch.device("cuda")
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "NariLabs/Dia-1.6B").
            device: The device to load the model onto.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="dia-v0_1.pth")
        return cls.from_local(config_path, checkpoint_path, device)

    @torch.inference_mode()
    def load_audio(self, audio_input: str | np.ndarray, input_sample_rate: int | None = None) -> torch.Tensor:
        """Loads audio from a file path or NumPy array and encodes it into DAC codes.

        Processes the audio waveform, resamples it to the DAC model's required
        sample rate, and uses the DAC encoder to obtain discrete audio codes.

        Args:
            audio_input: Either a string path to an audio file or a NumPy array
                         containing the audio waveform.
            input_sample_rate: The sample rate of the input audio if provided as a
                               NumPy array. If None and input is NumPy array, it's
                               *assumed* to be the DAC model's sample rate.
                               This argument is ignored if audio_input is a path.

        Returns:
            A tensor of shape [1, T_codes, C_codebooks] containing the discrete
            audio codes, ready to be used as an audio_prompt. T_codes is the
            number of timesteps in the code representation, and C_codebooks is
            the number of DAC codebooks.

        Raises:
            ValueError: If input is a NumPy array and input_sample_rate is not provided
                        and differs from the DAC model's sample rate (cannot be verified
                        without input_sample_rate).
            FileNotFoundError: If audio_input is a path and the file doesn't exist.
            RuntimeError: If audio loading or DAC encoding fails.
        """
        self.eval()  # Ensure model (including DAC) is in eval mode
        target_sr = self.dac_model.sample_rate

        if isinstance(audio_input, str):
            try:
                waveform, sr = torchaudio.load(audio_input)
            except FileNotFoundError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error loading audio file {audio_input}") from e
        elif isinstance(audio_input, np.ndarray):
            if input_sample_rate is None:
                # Assume the numpy array is already at the target sample rate
                sr = target_sr
                print(
                    f"Warning: input_sample_rate not provided for NumPy array input. "
                    f"Assuming audio is already at target sample rate {target_sr} Hz."
                )
            else:
                sr = input_sample_rate

            waveform = torch.from_numpy(audio_input).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        _, codes, _, _, _ = self.dac_model.encode(waveform, n_quantizers=None)
        return codes.transpose(1, 2)

    @torch.inference_mode()
    def decode_audio(self, codebooks: torch.Tensor) -> np.ndarray:
        """Decodes audio codebooks tensor into an audio waveform using the DAC model.

        Args:
            codebooks: A tensor of audio codes, typically the output of the
                       `generate` method. Expected shape [T_codes, C_codebooks]
                       or [1, T_codes, C_codebooks].

        Returns:
            A NumPy array representing the decoded audio waveform (mono).

        Raises:
            RuntimeError: If DAC decoding fails.
            ValueError: If the input tensor shape is unexpected.
        """
        self.eval()  # Ensure DAC is in eval mode

        # Ensure input has batch dim [1, T_codes, C]
        if codebooks.ndim == 2:
            codebooks = codebooks.unsqueeze(0)
        elif codebooks.ndim != 3 or codebooks.shape[0] != 1:
            raise ValueError(f"Unexpected shape for codebooks: {codebooks.shape}. Expected [T, C] or [1, T, C].")

        # DAC decode expects codes shape [B, C_codebooks, T_codes]
        codes_for_dac = codebooks.transpose(1, 2)

        try:
            # Mimic logic from dia/audio.py: codebook_to_audio -> decode
            # Need quantizer.from_codes -> model.decode
            # Note: dac v1.0.3 API might differ slightly. Assuming dac.decode takes codes directly.
            # If using older dac: audio_values = self.dac_model.quantizer.from_codes(codes_for_dac)
            #                       audio_values = self.dac_model.decode(audio_values[0]) # Index 0 is the quantized output
            audio_values = self.dac_model.decode(codes_for_dac)

        except Exception as e:
            raise RuntimeError(f"Error during DAC decoding: {e}") from e

        # Output shape is likely [B, 1, T_audio] or [B, T_audio]
        waveform_np = audio_values.squeeze().cpu().numpy()
        return waveform_np

    def save_audio(self, codebooks: torch.Tensor, output_path: str) -> None:
        """Decodes audio codebooks and saves the resulting waveform to a file.

        Args:
            codebooks: A tensor of audio codes, typically the output of the
                       `generate` method. Expected shape [T_codes, C_codebooks]
                       or [1, T_codes, C_codebooks].
            output_path: The path (including filename and extension, e.g., '.wav')
                         where the audio file will be saved.

        Raises:
            RuntimeError: If DAC decoding or file writing fails.
            ValueError: If the input codebook tensor shape is unexpected.
        """
        waveform_np = self.decode_audio(codebooks)
        try:
            sf.write(output_path, waveform_np, samplerate=self.dac_model.sample_rate)
            print(f"Audio saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving audio to {output_path}: {e}") from e

    def _encode_text_to_tokens(self, text: str) -> torch.Tensor:
        """Encodes a text string into a padded tensor of byte tokens.

        Handles speaker tokens ([S1], [S2]) by replacing them with special
        byte values. Pads or truncates the sequence to the length specified
        in the configuration (`config.data.text_length`).

        Args:
            text: The input text string.

        Returns:
            A tensor of shape [1, S] containing the encoded and padded byte tokens,
            where S is the configured text length. The tensor is on the model's device.
        """
        byte_text = text.encode("utf-8")

        for speaker_token, replacement in SPEAKER_TOKENS.items():
            byte_text = byte_text.replace(speaker_token, replacement)

        text_tokens = list(byte_text)

        current_len = len(text_tokens)
        max_len = self.config.data.text_length
        padding_needed = max_len - current_len

        if padding_needed <= 0:
            padded_text_np = np.array(text_tokens[:max_len], dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=self.config.data.text_pad_value,
            ).astype(np.uint8)

        return torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)  # [1, S]

    @torch.inference_mode()
    def run_encoder(self, cond_text: str) -> DecoderInferenceState:
        """Runs the encoder part of the model on the input text.

        Encodes the conditional text and creates an unconditional input (zeros).
        Runs both through the encoder and packages the results along with
        necessary parameters into a DecoderInferenceState object, ready for
        the decoder.

        Args:
            cond_text: The conditional input text string.

        Returns:
            A DecoderInferenceState object containing the encoder outputs,
            position information, masks, and initial state for the decoder.
        """
        self.eval()

        cond_src_BxS = self._encode_text_to_tokens(cond_text)
        unc_src_BxS = torch.zeros_like(cond_src_BxS)
        src_BxS = torch.cat([unc_src_BxS, cond_src_BxS], dim=0)

        state = EncoderInferenceState.new(self.config, cond_src_BxS)
        encoder_out = self.encoder(x_ids=src_BxS, state=state)

        return DecoderInferenceState.new(self.config, state, encoder_out)

    def _process_audio_prompt(self, audio_prompt: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        """Prepares an optional audio prompt for decoder input.

        If no prompt is provided, creates a default prompt containing only the
        BOS token. Otherwise, takes the provided audio token tensor, pads it
        to account for the maximum channel delay, and applies the audio delay
        pattern using precomputed indices.

        Args:
            audio_prompt: An optional tensor of audio tokens with shape [1, T_prompt, C],
                          where T_prompt is the prompt length and C is the number of channels.

        Returns:
            A tuple containing:
            - processed_audio_prompt (torch.Tensor): The prompt tensor ready for the
              decoder, padded and with delays applied. Shape [1, T_processed, C].
            - audio_prompt_len (int): The original length (T_prompt) of the audio
              prompt before padding and delay application. Returns 1 if no prompt was provided.
        """
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        apply_delay_precomp = (self._apply_delay_t_idx, self._apply_delay_indices)
        max_delay_pattern = max(self.config.data.delay_pattern)

        if audio_prompt is None:
            audio_prompt = torch.full(
                (1, 1, num_channels),
                fill_value=audio_bos_value,
                dtype=torch.long,
                device=self.device,
            )

        audio_prompt = audio_prompt.to(self.device)
        audio_prompt_len = audio_prompt.shape[1]
        audio_pad = torch.full(
            (1, max_delay_pattern, num_channels),
            fill_value=audio_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        audio_prompt = torch.cat([audio_prompt, audio_pad], dim=1)

        audio_prompt = apply_audio_delay(
            audio_BxTxC=audio_prompt,
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=apply_delay_precomp,
        )

        # audio_prompt:
        # BOS BOS BOS   a   b   c   d   e
        # BOS BOS   a   b   c   d   e PAD
        # BOS   a   b   c   d   e PAD PAD

        return audio_prompt, audio_prompt_len

    def _process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Processes the raw output tokens from the decoder loop.

        Reverts the audio delay pattern applied during generation using
        precomputed indices and removes padding introduced to handle the
        maximum delay.

        Args:
            output: The tensor of generated audio tokens directly from the
                    autoregressive loop, potentially containing delayed channels
                    and padding. Shape [T_raw, C].

        Returns:
            A tensor of processed audio tokens with delays reverted and padding
            removed. Shape [T_final, C].
        """
        audio_pad_value = self.config.data.audio_pad_value
        revert_delay_precomp = (self._revert_delay_t_idx, self._revert_delay_indices)
        max_delay_pattern = max(self.config.data.delay_pattern)

        output_len = output.shape[1]

        reverted_output = revert_audio_delay(
            audio_BxTxC=output, pad_value=audio_pad_value, precomp=revert_delay_precomp, T=output_len
        )

        return reverted_output.squeeze(0)[: output_len - max_delay_pattern, :]

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        audio_prompt: torch.Tensor | None = None,
        cfg_filter_top_k: int | None = 50,
        torch_compile_mode: str | None = "reduce-overhead",
    ) -> torch.Tensor:
        """Generates audio tokens based on text input and optional audio prompt.

        Performs end-to-end text-to-audio token generation using the encoder-decoder
        architecture, classifier-free guidance, and handling for multi-channel
        audio delays and optional audio prompting.

        Args:
            text: The input text string.
            cfg_scale: The scale factor for classifier-free guidance. Higher values
                       increase adherence to the text prompt.
            temperature: The temperature for sampling. Controls randomness.
            top_p: The cumulative probability for nucleus sampling.
            audio_prompt: Optional tensor of audio tokens [1, T_prompt, C] to use as a prompt.
            cfg_filter_top_k: Optional integer. If set, filters logits to top-k
                              before sampling during CFG calculation.
            torch_compile_mode: Optional string specifying the torch.compile mode
                                (e.g., "reduce-overhead", "max-autotune") or None
                                to disable compilation.

        Returns:
            A tensor of generated audio tokens with shape [T_final, C], where T_final
            is the length of the generated sequence and C is the number of channels.
            Delays have been reverted and padding removed.
        """
        self.eval()

        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(self.config.data.delay_pattern)
        delay_pattern = self.config.data.delay_pattern

        state = self.run_encoder(text)

        audio_prompt, audio_prompt_len = self._process_audio_prompt(audio_prompt)

        state.update_step(audio_prompt, d_step=audio_prompt_len)

        bos_countdown = max_delay_pattern
        eos_detected_channel_0 = False
        eos_countdown = -1
        processed_step = 0

        def step(
            decoder: Decoder,
            current_tokens_Bx1xC: torch.Tensor,
            state: DecoderInferenceState,
        ) -> torch.Tensor:
            logits_Bx1xCxV = decoder.decode_step(current_tokens_Bx1xC, state)

            logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]
            uncond_logits_CxV = logits_last_BxCxV[0, :, :]
            cond_logits_CxV = logits_last_BxCxV[1, :, :]

            logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)

            logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
            logits_CxV[1:, audio_eos_value] = -torch.inf

            pred_C = sample_next_token(
                logits_CxV.float(),
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
            )

            return pred_C

        if torch_compile_mode is not None:
            step_fn = torch.compile(step, mode=torch_compile_mode)
        else:
            step_fn = step

        while state.step < state.max_seq_len - 1:
            current_tokens_Bx1xC = state.generated_tokens[:, processed_step : state.step, :]
            pred_C = step_fn(self.decoder, current_tokens_Bx1xC, state, cfg_scale)

            if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
                eos_detected_channel_0 = True
                eos_countdown = max_delay_pattern

            if eos_detected_channel_0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        pred_C[i] = audio_eos_value
                    elif step_after_eos > d:
                        pred_C[i] = audio_pad_value

                eos_countdown -= 1

            state.update_step(pred_C.reshape((1, 1, -1)), apply_mask=bos_countdown > 0)
            bos_countdown = max(0, bos_countdown - 1)

            if eos_detected_channel_0 and eos_countdown == 0:
                break

        output = state.generated_tokens[0, audio_prompt_len : state.step, :]
        return self._process_output(output)
