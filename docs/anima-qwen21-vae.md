# Anima with Qwen Image 2.1 VAE

The model selector offers two independent model types:

| Selection | Config value | VAE | Image input | Typical latent shape |
| --- | --- | --- | --- | --- |
| Anima | `ANIMA` | Diffusers `AutoencoderKLQwenImage` | RGB | 16 channels, 8× spatial compression |
| Anima (qwen 2.1 vae) | `ANIMA_QWEN21_VAE` | OneTrainer `AutoencoderKLQwenImage21` | RGBA (or RGB when specified by the VAE config) | 64 channels, 16× spatial compression |

For the new type, choose a complete Diffusers checkpoint with a transformer adapted
to that VAE. Merely replacing the VAE in the original Anima checkpoint does not
adapt the transformer's input/output projections. OneTrainer checks the VAE class,
image channels, latent normalization, and transformer input/output channels on load.

OneTrainer contains the VAE implementation and inference pipeline. Loading a model
does not import Python files from its directory. The directory only supplies
configuration, tokenizers, and weights.

## Images and samples

For a four-channel VAE, existing alpha is preserved and RGB inputs receive opaque
alpha. All four channels are spatially transformed and encoded; brightness,
contrast, saturation, and hue modify only RGB. Training masks remain separate
single-channel tensors. Models without RGBA support retain their existing RGB
loading behavior.

PNG samples, concept previews, and cloud sample previews preserve alpha. JPEG
cannot store transparency, so RGBA samples saved as JPEG are composited on white.
Choose PNG to retain transparency.

Both Anima variants support fine-tuning and LoRA. The new variant uses a separate
cache under the existing `image` and `text` cache directories, which prevents reuse
of the original variant's incompatible latents. Multithreaded data loading
serializes VAE encoding because these VAEs maintain mutable frame caches.

Text encoder dropout is supported for both variants, with and without latent/text
caching. It replaces selected captions with the encoded empty prompt used for
classifier-free guidance. The frozen text encoder and conditioner prepare this
conditioning once before offloading; cached positive captions remain unchanged.

Debug mode writes decoded images, separate masks (when enabled), and their captions
to `<debug_dir>/dataloader/epoch-<number>`. Decoded images are always PNG, preserving
the fourth channel even when a source image is JPEG. These files show dataset
captions before the per-training-step caption dropout.

## Saving and resuming

Diffusers exports and internal training backups include the VAE configuration and
weights, transformer, text encoders/conditioner, scheduler, and tokenizers. The new
variant also exports the bundled pipeline and VAE source, so the saved package is
self-contained for external Diffusers use. External loading uses
`DiffusionPipeline.from_pretrained(path, trust_remote_code=True)`; OneTrainer always
uses its own bundled implementation.

The original-transformer format saves the transformer and text conditioner, as it
does for the original Anima. Use a Diffusers export or internal backup to save the
entire model including the VAE. LoRA exports contain adapters and require the
matching base model/VAE when loaded.

## Regression checks

Run with the project's installed dependencies:

```shell
python -m unittest discover -s tests -p test_rgba_images.py -v
python -m unittest discover -s tests -p test_anima_qwen21_vae.py -v
```

These tests use small real VAEs, transformers, tokenizers, and text encoders. They
cover both model types, RGBA/RGB compatibility, cached and uncached datasets,
masked training, an optimizer step, sampling, multithreaded VAE encoding,
caption dropout, RGB/RGBA debug exports over multiple epochs, Diffusers export/reload,
internal backup/resume, and LoRA export/reload.
