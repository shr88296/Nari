<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/pgdB5YRe" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Dataset on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- (Update) We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/pgdB5YRe) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## ⚡️ Quickstart

### Prerequisites

- **Python:** This project requires **Python 3.10**. We recommend using `pyenv` to manage Python versions. Ensure Python 3.10 is installed (`pyenv install 3.10`).
- **PyTorch:** The installation process will automatically install the appropriate PyTorch version for your system (CUDA for Linux/Windows with NVIDIA GPU, CPU or MPS for macOS).

### Install via pip

```bash
# Ensure you are using Python 3.10
# pyenv shell 3.10 # Example if using pyenv

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the package (this will also install dependencies)
pip install git+https://github.com/nari-labs/dia.git
# Or clone and install locally:
# git clone https://github.com/nari-labs/dia.git
# cd dia
# pip install -e .
```

### Run the Gradio UI

This will open a Gradio UI that you can work on locally.

```bash
# Make sure you are in the project directory ('dia') and the virtual environment is activated

# Recommended: Use Python 3.10
python3 app.py

# On macOS (Apple Silicon), you can specify the device:
# python3 app.py --device mps  # Use Metal Performance Shaders (Recommended for Apple Silicon)
# python3 app.py --device cpu   # Use CPU

# On Linux/Windows with NVIDIA GPU:
# python3 app.py --device cuda  # Use CUDA (Default if available)
```

*Note: The original instructions using `uv` might still work but could have environment inconsistencies. Using standard `pip` within a Python 3.10 virtual environment is the recommended approach for broader compatibility.* 

Note that the model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
You can keep speaker consistency by either adding an audio prompt (a guide coming VERY soon - try it with the second example on Gradio for now), or fixing the seed.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.

## ⚙️ Usage

### As a Python Library

```python
from dia.model import Dia
import torch # Import torch to specify device

# Determine device automatically or specify manually
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available(): # Check for MPS
    device = "mps"
else:
    device = "cpu"

# Load model specifying the device
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# Consider setting use_torch_compile=False on macOS for stability
output = model.generate(text, use_torch_compile=False, verbose=True)

model.save_audio("simple.mp3", output)
```

A pypi package and a working CLI tool will be available soon.

## 💻 Hardware and Inference Speed

Dia has been tested on NVIDIA GPUs (using CUDA) and can also run on macOS (Apple Silicon MPS or CPU) and potentially other CPUs.

- **NVIDIA GPU (Linux/Windows):** Requires PyTorch with CUDA support (tested with PyTorch 2.0+, CUDA 12.6). Provides the best performance.
- **macOS (Apple Silicon):** Can utilize Metal Performance Shaders (MPS) for GPU acceleration (`--device mps`). Performance will vary depending on the chip.
- **CPU:** Can run on CPU (`--device cpu`) on various platforms, but inference will be significantly slower.

*Note: The Triton library for optimized inference is only supported on Linux and Windows with NVIDIA GPUs.* 

The initial run will take longer as the Descript Audio Codec model weights also need to be downloaded.

These are the speed we benchmarked in RTX 4090 (CUDA).

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:-:|:-:|:-:|:-:|
| `bfloat16` | x2.1 | x1.5 | ~10GB |
| `float16` | x2.2 | x1.3 | ~10GB |
| `float32` | x1 | x0.9 | ~13GB |

*Performance on MPS and CPU will differ and depends on the specific hardware.* 

We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support for ARM architecture and MacOS.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/pgdB5YRe) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>
