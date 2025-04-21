from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .config import DiaConfig
from .layers import Decoder, Encoder


class Dia(nn.Module):
    """PyTorch Dia Model using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    @classmethod
    def from_local(cls, config_path: Path, checkpoint_path: Path, device: torch.device) -> "Dia":
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        model = cls(config)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")
        model.to(device)
        model.eval()
        return model

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> "Dia":
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="dia-v0_1.pth")
        return cls.from_local(config_path, checkpoint_path, device)
