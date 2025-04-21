# Nari TTS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<!-- Add other badges here: PyPI version, Hugging Face model, etc. -->

A Text-to-Speech model implementation, refactored into a Python library with CLI and Gradio interfaces.

_(Add a more detailed description of the model and its capabilities here)_.

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

### Installation

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

pypi package coming soon.

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
    temperature=1.2,
    top_p=0.95,
    use_cfg_filter=True,
    device=device,
    cfg_filter_top_k=35,
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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## TODO / Future Work

- Refine Hugging Face Hub integration (e.g., `push_to_hub`).
- Improve documentation and add API reference.
- Add unit and integration tests.
- Optimize inference speed.
- Publish to PyPI.
- Enhance Gradio UI (e.g., add examples, status updates).
- Clean up remaining `src/` directory contents.

## Contributing

_(Add contribution guidelines if desired)._
