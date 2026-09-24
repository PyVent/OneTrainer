import json
import os

ANIMA_LATENT_CHANNELS = 16
MAX_SAFETENSORS_HEADER_SIZE = 16 * 1024 * 1024


def _single_file_latent_channels(source: str) -> int | None:
    try:
        with open(source, "rb") as file:
            size_bytes = file.read(8)
            if len(size_bytes) != 8:
                raise ValueError("incomplete safetensors header")
            header_size = int.from_bytes(size_bytes, "little")
            if not 0 < header_size <= MAX_SAFETENSORS_HEADER_SIZE:
                raise ValueError("invalid safetensors header size")
            header = json.loads(file.read(header_size))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Anima VAE file is not a valid safetensors checkpoint: {source}") from error

    for key in ("decoder.conv_in.weight", "decoder.conv1.weight"):
        shape = header.get(key, {}).get("shape", [])
        if len(shape) >= 2 and isinstance(shape[1], int):
            return shape[1]
    return None


def validate_anima_vae_source(source: str) -> None:
    """Reject unsupported or incompatible single-file Anima VAE overrides early."""
    if not source or not os.path.isfile(source):
        return

    if source.lower().endswith(".safetensors"):
        channels = _single_file_latent_channels(source)
        if channels is not None and channels != ANIMA_LATENT_CHANNELS:
            raise ValueError(
                f"Anima VAE has {channels} latent channels, but the Anima transformer expects "
                f"{ANIMA_LATENT_CHANNELS}. This VAE cannot replace the Anima VAE directly: {source}"
            )

    raise ValueError(
        "Anima VAE override must be a Diffusers model directory or repository "
        f"containing config.json and model weights, not a single file: {source}"
    )
