# Pilot AI

Pilot AI is a local, from-scratch language-model training project. The current
track trains a small Llama-style causal LM and exports it to GGUF so it can be
loaded by tools such as LM Studio, llama.cpp, and Ollama.

This repository does not include downloaded datasets, checkpoints, exported
GGUF files, or secret tokens.

## Current Version: Pilot 0.0.1

- Architecture: Llama-compatible causal language model
- Size: about 98M parameters
- Tokenizer: GPT-2 BPE tokenizer saved locally with a Pilot chat template
- Training data: English/Spanish web text plus repeated seed chat examples
- Export target: GGUF Q8_0 or F16
- Vision stage: SigLIP encoder scaffold for a later multimodal adapter

The model is trained from random weights. It does not use a pretrained base
language model.

## Version Status

Pilot uses semantic experimental versions. `0.0.0` is the project scaffold.
`0.0.1` is the first exported from-scratch GGUF baseline. It loads in LM Studio,
but it is not considered a usable chat model yet: short prompts still produce
unstable text. See `experiments/0.0.1.md`.

## Quick Start

Create and activate a Python environment, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Set up the tokenizer and prepare the balanced chat/text dataset:

```powershell
python scripts\download_bilingual_corpus.py --en-gb 1 --es-gb 1 --out-dir data\raw --overwrite
python scripts\setup_pilot_0_0_1_tokenizer.py
python scripts\prepare_pilot_0_0_1_data.py --out-dir data\pilot_0_0_1_tokens_chatmix --max-gb-per-language 0.35 --chat-repeats 500000 --max-val-tokens 1000000
```

Train Pilot 0.0.1:

```powershell
python scripts\train_pilot_0_0_1.py --config configs\pilot_0_0_1_llama_chat_fast.json
```

The trainer prints loss, tokens per second, average tokens per second, GPU
temperature, progress, tokens trained, elapsed time, and ETA.

Resume from the latest checkpoint:

```powershell
python scripts\train_pilot_0_0_1.py --config configs\pilot_0_0_1_llama_chat_fast.json --resume runs\pilot_0_0_1_llama_chat_fast\hf_last
```

Export the latest checkpoint to GGUF:

```powershell
python scripts\export_pilot_0_0_1_gguf.py --hf-dir runs\pilot_0_0_1_llama_chat_fast\hf_last --outfile exports\pilot_0_0_1-llama-chat-q8_0.gguf --outtype q8_0 --write-modelfile
```

The default output is:

```text
exports\pilot_0_0_1-llama-chat-q8_0.gguf
```

## LM Studio / Ollama

After exporting GGUF, copy the `.gguf` file into LM Studio or reference it from
an Ollama `Modelfile`. The export script can also write:

```text
exports\Modelfile.pilot-0.0.1
```

Because Pilot is trained from scratch and is very small, it should be treated as
an experimental local model. It will not behave like a mature instruction-tuned
assistant without more supervised chat data and additional tuning.

## Vision

The vision stage downloads a SigLIP encoder in fp16:

```powershell
python scripts\setup_pilot_0_0_1_vision.py --config configs\pilot_0_0_1_vision_siglip.json
```

This prepares the vision encoder only. To make Pilot accept images in a chat UI,
the next stage is training a projector/mmproj-style adapter on image-caption or
image-chat data and exporting it in a format supported by the target runtime.

## Project Layout

```text
configs/      Training and vision configs
pilot_lm/     Shared Python helpers
scripts/      Data preparation, training, export, and setup scripts
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
