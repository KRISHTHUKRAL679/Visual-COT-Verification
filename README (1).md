# VisVerify — Visual CoT Verification

## Model
Use this 

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

The code loads it locally through `transformers` with 4-bit quantization. 



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
