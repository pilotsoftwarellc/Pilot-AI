# Pilot AI

Pilot AI is a local, from-scratch language-model training project. The current
track trains a small Llama-style causal LM and exports it to GGUF so it can be
loaded by tools such as LM Studio, llama.cpp, and Ollama.

This repository does not include downloaded datasets, checkpoints, exported
GGUF files, or secret tokens.

## Current Development Target: Pilot 0.0.2

- Architecture: Llama-compatible causal language model
- Size: about 103M parameters
- Tokenizer: custom 16k byte-level BPE trained on English/Spanish data
- Training data: English/Spanish pretrain plus supervised chat tuning
- Export target: GGUF Q8_0 or F16
- Vision stage: SigLIP encoder scaffold for a later multimodal adapter

The model is trained from random weights. It does not use a pretrained base
language model.

## Version Status

Pilot uses semantic experimental versions.

- `0.0.0`: project scaffold.
- `0.0.1`: first exported GGUF baseline. It loads in LM Studio but fails as a
  useful chat model. See `experiments/0.0.1.md`.
- `0.0.2-dev`: current training recipe. Goal: keep the model near 100M params
  while fixing tokenizer size, training volume, supervised chat tuning, and eval.

## Quick Start

Create and activate a Python environment, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Set up the 0.0.2 tokenizer and datasets:

```powershell
python scripts\download_bilingual_corpus.py --en-gb 1 --es-gb 1 --out-dir data\raw --overwrite
python scripts\setup_pilot_0_0_2_tokenizer.py
python scripts\prepare_pilot_0_0_2_pretrain_data.py --max-gb-per-language 2.0 --chat-repeats 20000
python scripts\prepare_pilot_0_0_2_sft_data.py --repeats 2500
```

Benchmark before a full run:

```powershell
python scripts\train_pilot_0_0_2.py --config configs\pilot_0_0_2_benchmark.json
```

Train Pilot 0.0.2 in two stages:

```powershell
python scripts\train_pilot_0_0_2.py --config configs\pilot_0_0_2_pretrain.json
python scripts\train_pilot_0_0_2.py --config configs\pilot_0_0_2_sft.json
```

The trainer prints loss, tokens per second, average tokens per second, GPU
temperature, progress, tokens trained, elapsed time, and ETA.

Resume from the latest checkpoint:

```powershell
python scripts\train_pilot_0_0_2.py --config configs\pilot_0_0_2_pretrain.json --resume runs\pilot_0_0_2_pretrain\hf_last
python scripts\train_pilot_0_0_2.py --config configs\pilot_0_0_2_sft.json --resume runs\pilot_0_0_2_sft\hf_last
```

Run the quality gate before exporting:

```powershell
python scripts\eval_pilot_0_0_2.py --hf-dir runs\pilot_0_0_2_sft\hf_last
```

Export the latest checkpoint to GGUF:

```powershell
python scripts\export_pilot_0_0_2_gguf.py --hf-dir runs\pilot_0_0_2_sft\hf_last --outfile exports\pilot_0_0_2-llama-chat-q8_0.gguf --outtype q8_0 --write-modelfile
```

The default output is:

```text
exports\pilot_0_0_2-llama-chat-q8_0.gguf
```

## LM Studio / Ollama

After exporting GGUF, copy the `.gguf` file into LM Studio or reference it from
an Ollama `Modelfile`. The export script can also write:

```text
exports\Modelfile.pilot-0.0.2
```

Because Pilot is trained from scratch and is very small, it should be treated as
an experimental local model. It will not behave like a mature instruction-tuned
assistant without more supervised chat data and additional tuning.

## Vision

Vision is deferred until the text model passes the 0.0.2 quality gate. Adding
images before basic chat works would make the debugging problem harder.

## Project Layout

```text
configs/      Training configs
pilot_lm/     Shared Python helpers
scripts/      Data preparation, training, export, and setup scripts
```

Ignored local folders:

```text
data/         Raw datasets and token bins
runs/         Training checkpoints
exports/      GGUF exports and Modelfiles
models/       Optional downloaded local models
tokenizers/   Generated local tokenizer files
external/     llama.cpp or other external clones
_secrets/     Local tokens and private files
```

## License

MIT. See `LICENSE`.
