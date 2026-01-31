# LLM Distillation Practice

This folder is for practicing LLM distillation using a pretrained DeepSeek model. Follow the steps below to get started:

## Steps to Follow:
1. **Download the Pretrained DeepSeek Model**: 
   - Ensure you have the model files saved in this directory.

2. **Set Up Your Environment**:
   - Create a virtual environment (optional but recommended).
   - Install necessary packages (e.g., TensorFlow, PyTorch, etc.).

3. **Load the Model**:
   - Use the appropriate library to load the pretrained model.

4. **Prepare Your Dataset**:
   - Format your dataset for distillation.

5. **Implement Distillation Logic**:
   - Write the code to perform distillation using the loaded model and prepared dataset.

6. **Run Experiments**:
   - Execute your distillation process and evaluate the results.

7. **Analyze Results**:
   - Review the performance of the distilled model and make adjustments as necessary.

---

## Recommended packages ✅
Install the packages below for downloading, loading, and distilling models (GPU recommended):

```bash
pip install -r requirements.txt
```

Key packages included in `requirements.txt`:
- `huggingface-hub` (download, auth)
- `transformers` (model loading)
- `torch` (training/inference)
- `accelerate` (multi-GPU / mixed precision)
- `safetensors`, `sentencepiece`, `tokenizers` (fast and safe tokenizers/weights)
- `bitsandbytes`, `peft` (8-bit training / LoRA) — optional but useful for efficient experiments

## Which files to download from the Hugging Face model page ⚖️
When you're on the model page (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`), prefer these files for use as the *teacher* model:
- `pytorch_model.bin` or `model.safetensors` (full or fp16 weights) — choose **safetensors** if offered.
- `config.json` and any `tokenizer` files (`tokenizer.json`, `tokenizer_config.json`, vocab files).

Avoid using quantized/ggml files as the teacher for distillation — those are intended for CPU inference and may degrade the teacher signal.

## How to download the model ✅
1. Authenticate to HF (recommended):
```bash
huggingface-cli login  # or set HUGGINGFACE_HUB_TOKEN env var
```

2. Use the provided helper script to snapshot the repo into `llm-distillation/models/`:
```bash
python scripts/download_model.py
```

Alternatively, download manually with Python:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', cache_dir='llm-distillation/models/DeepSeek-R1-Distill-Qwen-1.5B')
```

---

## Distillation script (quick start) 🔧
I added a lightweight distillation script at `scripts/distill.py`. It implements a simple training loop combining cross-entropy and KL-based logits distillation. The script is MPS-aware (uses Apple M1/M2/M3/M4 `mps` backend when available) and avoids fp16 on MPS by default.

Quick examples (run inside your `llm-distill` conda env):

- Minimal quick test using a tiny dataset (local text file or HF dataset):
```bash
conda activate llm-distill
python scripts/distill.py \
  --teacher-path llm-distillation/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --student-model gpt2 \
  --text-file /path/to/small_text_corpus.txt \
  --max-samples 500 \
  --batch-size 2 \
  --epochs 1 \
  --output-dir llm-distillation/out_student_test
```

- Use a HF dataset for quick runs (wikitext small):
```bash
python scripts/distill.py --teacher-path <teacher_path> --student-model gpt2 --dataset "wikitext/wikitext-2-raw-v1" --max-samples 1000 --batch-size 2
```

Notes:
- On Apple Silicon (M1/M2/M3/M4), the script will pick `mps` if available. MPS float16 support is limited — the script uses fp32 for MPS to be safe. ✅
- If you have CUDA GPUs and want FP16 training, add `--force-fp32` to override FP16 and force FP32; remove it to allow fp16 on CUDA.

---

(Existing instructions remain below.)
## Notes / tips 💡
- For distillation, using the teacher in full precision (fp32) or fp16 is best. If you must save memory, fp16 is acceptable. Do not use quantized weights as the teacher.
- If you plan to do memory-efficient fine-tuning or distillation on limited hardware, consider `bitsandbytes` + `accelerate` and training recipes like LoRA or progressive distillation.

---

## Additional Resources:
- Model page: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Hugging Face docs: https://huggingface.co/docs
- Distillation overview: https://www.semanticscholar.org/topic/Knowledge-Distillation