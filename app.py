import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import dac
import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.inference import (
    codebook_to_audio,
    generate,
    load_model_and_config,
)


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--config", type=str, default="./config.json", help="Path to model config file")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./dia-v0_1.pth",
    help="Path to model checkpoint file",
)
parser.add_argument("--dac_model_type", type=str, default="44khz", help="DAC model type (e.g., 44khz)")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()

CONFIG_PATH = Path(args.config)
CHECKPOINT_PATH = Path(args.checkpoint)

if not CONFIG_PATH.exists():
    # Adding print statement for clarity in logs if run non-interactively
    print(f"Error: Config file not found at {CONFIG_PATH}")
    raise FileNotFoundError(f"Error: Config file not found at {CONFIG_PATH}")
if not CHECKPOINT_PATH.exists():
    print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
    raise FileNotFoundError(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")

# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Use the function from inference.py
    model, config = load_model_and_config(CONFIG_PATH, CHECKPOINT_PATH, device)
    # Ensure model is in eval mode after loading
    model.eval()
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

# Placeholder for DAC model - load lazily
dac_model = None
dac_model_path = None
print(f"DAC model type set to: {args.dac_model_type}")


def ensure_dac_model_loaded():
    """Loads the DAC model if it hasn't been loaded yet."""
    global dac_model, dac_model_path, device, args
    if dac_model is None:
        print("Loading DAC model...")
        try:
            dac_model_path = dac.utils.download(model_type=args.dac_model_type)
            # Ensure model is loaded to the correct device
            dac_model = dac.DAC.load(dac_model_path).to(device)
            dac_model.eval()  # Set DAC to eval mode as well
            print(f"DAC model loaded successfully from {dac_model_path} to {device}.")
        except Exception as e:
            print(f"Error loading DAC model ({args.dac_model_type}): {e}")
            # Make sure dac_model remains None if loading fails
            dac_model = None
            raise gr.Error(f"Failed to load DAC model: {e}")  # Raise Gradio error for UI feedback
    return dac_model


# --- Gradio Inference Function ---
# @torch.inference_mode() # Apply inference mode decorator for efficiency
def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    cfg_filter_top_k: int,
    progress=gr.Progress(track_tqdm=True),  # Add progress tracking
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, config, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    # Default output: standard sample rate (use DAC's later), silent audio
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        # Ensure model is on the correct device before inference
        model.to(device)

        # 1. Prepare Text Input File
        # Use NamedTemporaryFile for automatic cleanup potential, but manage explicitly
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f_txt:
            f_txt.write(text_input)
            temp_txt_file_path = f_txt.name
        print(f"Created temporary text file: {temp_txt_file_path}")

        # 2. Prepare Audio Prompt Input File (if provided)
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Ensure DAC model is loaded if a valid prompt is provided
                _ = ensure_dac_model_loaded()  # Load DAC if needed, ignore return val for now

                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(audio_data)  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation
        # Pass the currently loaded DAC model instance (if loaded) to generate
        # This avoids generate trying to load it again unnecessarily
        dac_model_instance = dac_model  # Use the globally loaded instance

        print("\n--- Starting Generation ---")
        print(f"Params: max_tokens={max_new_tokens}, cfg={cfg_scale}, temp={temperature}, top_p={top_p}")
        start_time = time.time()

        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            generated_codes = generate(
                model=model,
                config=config,
                txt_file_path=temp_txt_file_path,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=50,  # Ensure int
                device=device,
                use_torch_compile=False,  # Keep False for Gradio stability
                audio_prompt_path=prompt_path_for_generate,
                dac_model=dac_model_instance,  # Pass the loaded dac model instance
            )

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        print(f"Generated codes shape: {generated_codes.shape if generated_codes is not None else 'None'}")

        # 4. Convert Codes to Audio
        if generated_codes is not None and generated_codes.numel() > 0:
            print("Converting generated codes to audio...")
            # Ensure DAC model is loaded for conversion step
            current_dac_model = ensure_dac_model_loaded()
            if not current_dac_model:  # Check if loading failed earlier
                raise gr.Error("DAC model is required for audio conversion but failed to load.")

            # Ensure codes have the expected shape [T, C] for codebook_to_audio's input transpose
            if generated_codes.dim() == 3 and generated_codes.shape[0] == 1:
                codes_for_dac = generated_codes.squeeze(0)
            elif generated_codes.dim() == 2:
                codes_for_dac = generated_codes
            else:
                raise ValueError(f"Unexpected shape for generated_codes: {generated_codes.shape}")

            # codebook_to_audio expects codes shape [C, T] after transpose
            # It returns audio tensor shape [1, T_audio]
            with torch.inference_mode():  # Use inference mode for DAC conversion too
                audio_tensor = codebook_to_audio(
                    codes_for_dac.transpose(0, 1),  # Transpose T,C -> C,T
                    current_dac_model,
                    config.data.delay_pattern,
                    C=config.data.channels,
                )

            output_audio_np = audio_tensor.cpu().float().numpy().squeeze()  # Ensure float type
            # Get sample rate from the loaded DAC model
            output_sr = current_dac_model.sample_rate
            output_audio = (output_sr, output_audio_np)
            print(f"Audio conversion successful. Output shape: {output_audio_np.shape}, Sample Rate: {output_sr}")

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}")
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}")

    return output_audio


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "Enter text for speech synthesis here."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")
    gr.Markdown(f"**Model:** `{CHECKPOINT_PATH.name}` | **Config:** `{CONFIG_PATH.name}` | **Device:** `{device}`")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",  # Returns (sr, np.ndarray)
                # Example: Add waveform visualization options if desired
                # waveform_options=gr.WaveformOptions(waveform_color="#01C6FF", waveform_progress_color="#0066B2", show_duration=True)
            )
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=2600,
                    value=config.data.audio_length,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=0.1,
                    maximum=1.5,
                    value=1.2,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",  # Expects (sr, np.ndarray) from function
                autoplay=False,
                # interactive=False # Output shouldn't be interactive as input
            )
            # Add status or log output if desired
            # status_output = gr.Textbox(label="Status", interactive=False, lines=3)

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
        ],
        outputs=[audio_output],  # Add status_output here if using it
        api_name="generate_audio",
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "assets/example_prompt.wav"  # Adjust if needed
    examples_list = [
        [
            "Hello, this is a test of the Nari text to speech system.",
            None,
            1000,
            3.0,
            1.0,
            0.98,
            True,
            50,
        ],
        [
            "You can also provide an audio prompt to guide the style. This example uses a prompt.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            1500,
            3.5,
            1.0,
            0.98,
            True,
            50,
        ],
    ]
    # Filter out examples with missing prompt files
    examples_list = [ex for ex in examples_list if ex[1] is not None or not Path(example_prompt_path).exists()]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
            ],
            outputs=[audio_output],  # Add status_output here if using it
            fn=run_inference,  # Function to run for examples
            cache_examples=False,  # Caching might be slow or complex with models
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    # Add server_name="0.0.0.0" to listen on all interfaces if needed (e.g., for Docker)
    demo.launch(share=True)
