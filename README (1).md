# VisVerify — Visual CoT Verification

## Model
Use this on an RTX 4060 8GB:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

The code loads it locally through `transformers` with 4-bit quantization. There
is no Anthropic/OpenAI/model API call in this project.

## Install
Create/use your Python env from the repo root, then:

```bash
pip install Pillow datasets accelerate bitsandbytes qwen-vl-utils
pip install git+https://github.com/huggingface/transformers
```

The model will download into the normal Hugging Face cache, usually:

```text
~/.cache/huggingface/hub/
```

You can override the model:

```bash
export VISVERIFY_QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

## Dataset
The loader streams only 10 rows from:

```text
multimodal-reasoning-lab/Zebra-CoT
3D Visual Reasoning - Multi-Hop Objects Counting / train
```

It caches only these fields:

- `Question`
- `Text Reasoning Trace`
- `Final Answer`
- `problem_image_1`

Local cache:

```text
visverify/data/zebra_3d_counting_10/
```

## Run
```bash
python3 visverify/run_eval.py
```

## Current Pipeline
```text
problem_image_1 + Text Reasoning Trace
  -> split trace into step claims
  -> parse each claim into typed facts when possible
  -> use Qwen2.5-VL to detect/count/localize objects
  -> run bbox/count/spatial/attribute asserts
  -> score each assert with perceptual confidence
  -> cap each step by previous-step confidence
  -> update registry belief state
```

If a trace step is too free-form for the parser, VisVerify asks the local Qwen
model a yes/no visual question for that step.
