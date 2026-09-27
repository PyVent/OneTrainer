# Model exports and training checkpoints

These outputs serve different purposes:

| Output | Contents | Where to create it |
| --- | --- | --- |
| LoRA adapter | Trained adapter weights; requires the base model | Normal LoRA training export, or Convert models → LoRA → Export LoRA adapter only |
| Merged model | Base weights with the LoRA update applied | Convert models → LoRA → Merge LoRA into base model |
| Training checkpoint | Weights plus optimizer, scheduler and training progress | Checkpoints and exports → Save training checkpoint |

To merge a saved LoRA, select its file, the exact base model it was trained on,
and **Merge LoRA into base model**. Choose **Complete model — Diffusers directory**
for a full pipeline. A transformer-only format contains only the denoiser and
still needs the other components. The input and output paths must be different.
For Anima and Anima with Qwen 2.1 VAE, the full pipeline retains the corresponding
VAE, text encoder and text conditioner.

The merge accepts standard LoRA and DoRA weights, including fused attention
projections. Other PEFT types are rejected explicitly; they are not silently
exported as unchanged base weights. Merging performs the update in float32 before
casting back to the requested export precision. It does not modify the input files.

CLI equivalent:

```powershell
.\venv\Scripts\python.exe scripts\convert_model.py --model-type ANIMA_QWEN21_VAE --training-method LORA --merge-lora --input-name adapter.safetensors --base-model-name path/to/base --output-model-format DIFFUSERS --output-model-destination path/to/merged --output-dtype BFLOAT_16
```

**Resume from latest training checkpoint** restores training state from the newest
complete checkpoint in `<workspace>/backup`. The directory and configuration key
retain their existing names for compatibility. Loading a standalone LoRA file
initializes training from those weights; it cannot restore optimizer state that
the file does not contain.

Standalone generation disables the training compile option. A manual sample made
during an active training run uses that run's already configured model. Preview
images are scaled to the generation window; the saved image keeps its full resolution.
