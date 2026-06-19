# Pilot AI

Pilot AI is a local, from-scratch language-model training project. The current
track trains a small Llama-style causal LM and exports it to GGUF so it can be
loaded by tools such as LM Studio, llama.cpp, and Ollama.

This repository does not include downloaded datasets, checkpoints, exported
GGUF files, or secret tokens.

## Current Track: Pilot 1.1

- Architecture: Llama-compatible causal language model
- Size: about 98M parameters
- Tokenizer: GPT-2 BPE tokenizer saved locally with a Pilot chat template
- Training data: English/Spanish web text plus repeated seed chat examples
- Export target: GGUF Q8_0 or F16
- Vision stage: SigLIP encoder scaffold for a later multimodal adapter

The model is trained from random weights. It does not use a pretrained base
language model.

## Quick Start

Create and activate a Python environment, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Set up the tokenizer and prepare the balanced chat/text dataset:

```powershell
.\start_pilot_1_1_setup_visible.cmd
```

Train Pilot 1.1 in a visible CMD window:

```powershell
.\start_pilot_1_1_train_visible.cmd
```

The trainer prints loss, tokens per second, average tokens per second, GPU
temperature, progress, tokens trained, elapsed time, and ETA. If a checkpoint
already exists at `runs\pilot_1_1_llama_chat_fast\hf_last`, the launcher resumes
from it automatically.

Export the latest checkpoint to GGUF:

```powershell
.\start_pilot_1_1_export_visible.cmd
```

The default output is:

```text
exports\pilot_1_1-llama-chat-q8_0.gguf
```

## LM Studio / Ollama

After exporting GGUF, copy the `.gguf` file into LM Studio or reference it from
an Ollama `Modelfile`. The export script can also write:

```text
exports\Modelfile.pilot-1.1
```

Because Pilot is trained from scratch and is very small, it should be treated as
an experimental local model. It will not behave like a mature instruction-tuned
assistant without more supervised chat data and additional tuning.

## Vision

The vision stage downloads a SigLIP encoder in fp16:

```powershell
.\start_pilot_1_1_vision_setup_visible.cmd
```

This prepares the vision encoder only. To make Pilot accept images in a chat UI,
the next stage is training a projector/mmproj-style adapter on image-caption or
image-chat data and exporting it in a format supported by the target runtime.

## Legacy Pilot 1.0

The repository still includes the earlier byte-level GPT training scripts. Those
were useful for experimentation, but Pilot 1.1 is the preferred path for
runtime compatibility because it exports as a Llama-compatible GGUF model.

## Project Layout

```text
configs/      Training and vision configs
pilot_lm/     Shared Python helpers
scripts/      Data preparation, training, export, and setup scripts
*.cmd         Windows launchers for visible training/setup/export
```

Ignored local folders:

```text
data/         Raw datasets and token bins
runs/         Training checkpoints
exports/      GGUF exports and Modelfiles
models/       Downloaded vision encoders
tokenizers/   Generated local tokenizer files
external/     llama.cpp or other external clones
_secrets/     Local tokens and private files
```

## License

MIT. See `LICENSE`.
