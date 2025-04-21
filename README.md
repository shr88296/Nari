# Dia

[![Join us on Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord)](https://discord.gg/pgdB5YRe)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<!-- Add other badges here: PyPI version, Hugging Face model, etc. -->

Dia is a 1.6B parameter speech generation model created by Nari Labs. Dia can generate highly realistic dialogue from a transcript. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/NariLabs/Dia).

[Demo Page](https://yummy-fir-7a4.notion.site/dia-demo) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

## Features

- **Python Library:** Core functionalities accessible via `import nari_tts`.
- **Command-Line Interface (CLI):** Generate audio from text using `scripts/infer.py`.
- **Gradio Web UI:** Interactive demo interface via `app/app.py`.
- **Hugging Face Hub Integration:** Load models directly from the Hub (`buttercrab/nari-tts` placeholder).
- **Docker Support:** Run the CLI or Gradio app in isolated containers.
  - `Dockerfile.cli`
  - `Dockerfile.app`
  - `docker-compose.yml` (for Gradio app)
- **Audio Prompting:** Guide speech style using an audio prompt (optional).

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.6.0 (check compatibility with your CUDA version if applicable)
- `uv` (optional, for faster dependency installation: `pip install uv`)

### Installing the Library

**1. From Source (Recommended for Development):**

Clone the repository and install in editable mode:

```bash
# Clone the repository (replace with your actual repo URL)
git clone https://github.com/your-username/nari-dialogue.git
cd nari-dialogue

# Install using pip (uses pyproject.toml)
pip install -e .

# Or install using uv (faster)
uv pip install -e .
```

**2. From PyPI (Once Published):**

```bash
pip install nari-tts
```

**3. Installing Dependencies Separately:**

You can install dependencies using `uv` or `pip` with the provided files:

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

## Usage

### As a Python Library

```python
import torch
import dac
import soundfile as sf
from pathlib import Path
from nari_tts import load_model_from_hub, generate, codebook_to_audio

# --- Configuration ---
REPO_ID = \"buttercrab/nari-tts\" # Replace with actual Hub ID
DAC_MODEL_TYPE = \"44khz\"
OUTPUT_FILENAME = \"library_output.wav\"
INPUT_TEXT = \"This audio was generated using the Nari TTS library.\"

# --- Device ---
device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
print(f\"Using device: {device}\")

# --- Load Models ---
print(f\"Loading Nari model from {REPO_ID}...\")
model, config = load_model_from_hub(REPO_ID, device=device)

print(f\"Loading DAC model ({DAC_MODEL_TYPE})...\")
dac_model_path = dac.utils.download(model_type=DAC_MODEL_TYPE)
dac_model = dac.DAC.load(dac_model_path).to(device)
dac_model.eval()
dac_sample_rate = dac_model.sample_rate

# --- Generate ---
print(f\"Generating audio for: '{INPUT_TEXT}'\")
generated_codes = generate(
    model=model,
    config=config,
    text=INPUT_TEXT,
    max_tokens=config.data.audio_length, # Or specify a number
    cfg_scale=3.0,
    temperature=1.0,
    top_p=0.95,
    use_cfg_filter=True,
    device=device,
    cfg_filter_top_k=50,
    # audio_prompt_path=\"path/to/prompt.wav\", # Optional
    dac_model=dac_model
)

# --- Decode and Save ---
if generated_codes.numel() > 0:
    print(\"Converting codes to audio...\")
    audio_tensor = codebook_to_audio(
        generated_codes=generated_codes.transpose(0, 1), # Needs [C, T]
        dac_model=dac_model,
        delay_pattern=config.data.delay_pattern,
        C=config.data.channels,
    )
    audio_np = audio_tensor.cpu().float().numpy().squeeze()
    sf.write(OUTPUT_FILENAME, audio_np, dac_sample_rate)
    print(f\"Audio saved to {OUTPUT_FILENAME}\")
else:
    print(\"Generation failed.\")

```

### Command-Line Interface (CLI)

The CLI script `scripts/infer.py` allows generation from the terminal.

**Basic Usage (Loading from Hub):**

```bash
python scripts/infer.py \"Your input text goes here.\" \
    --repo-id buttercrab/nari-tts \
    --output generated_speech.wav
```

**Loading from Local Files:**

```bash
python scripts/infer.py \"Text for local model.\" \
    --local-paths \
    --config path/to/your/config.json \
    --checkpoint path/to/your/nari_v0.pth \
    --output local_output.wav
```

**With Audio Prompt:**

```bash
python scripts/infer.py \"Generate speech like this prompt.\" \
    --repo-id buttercrab/nari-tts \
    --audio-prompt path/to/your/prompt.wav \
    --output prompted_output.wav
```

**See all options:**

```bash
python scripts/infer.py --help
```

### Gradio Web UI

The Gradio app provides an interactive interface.

**Running Locally:**

```bash
# Load from Hub (replace repo-id)
python app/app.py --repo-id buttercrab/nari-tts

# Load from local files
python app/app.py --local-paths --config path/to/config.json --checkpoint path/to/checkpoint.pth
```

Access the UI in your browser (usually at `http://127.0.0.1:7860`).

### Docker

**1. Build the Images:**

```bash
# Build CLI image
docker build -t nari-tts-cli:latest -f Dockerfile.cli .

# Build App image
docker build -t nari-tts-app:latest -f Dockerfile.app .
```

**2. Run CLI Container:**

```bash
docker run --rm -v \"$(pwd)/output:/app/output\" nari-tts-cli:latest \
    \"Generating audio inside a Docker container.\" \
    --repo-id buttercrab/nari-tts \
    --output /app/output/docker_output.wav

# Mount ~/.cache/huggingface to reuse downloads:
# docker run --rm -v \"$(pwd)/output:/app/output\" -v \"~/.cache/huggingface:/root/.cache/huggingface\" nari-tts-cli:latest ...
```

_(Note: Adjust volume mounts as needed for your OS and paths.)_

**3. Run Gradio App Container (using Docker Compose):**

This is the easiest way to run the Gradio app with proper port mapping and volume mounts.

```bash
docker compose up
```

This will build the `nari-tts-app` image if it doesn't exist and start the container.
Access the UI at `http://localhost:7860`.

To stop the service:

```bash
docker compose down
```

## Model

_(Add details about the model architecture, training data, expected quality, and link to the Hugging Face model card here)_.

## Disclaimer

This project offers a high-fidelity speech generation model intended solely for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## TODO / Future Work

- Optimize inference speed.
- Add quantization for memory efficiency.

## Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/pgdB5YRe) for discussions.

## Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636) and [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/).
- "Nari" is a pure Korean word for lily, hence the logo.
