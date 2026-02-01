# LLM Distillation Environment Setup

This project supports multiple environment setup methods. Choose the one that fits your needs.

---

## Option 1: Conda Environment (Recommended)

### Create and activate environment:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate llm-distill
```

### For Apple Silicon (M1/M2/M3/M4):
```bash
# Create environment (PyTorch will use MPS backend)
conda env create -f environment.yml

# Activate
conda activate llm-distill

# Note: Remove pytorch-cuda line from environment.yml before creating
```

### Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Option 2: pip + virtualenv

### Create virtual environment:
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verify Setup

After setting up your environment, verify it works:

```bash
# Test model download
python scripts/download_deepseek_model.py

# Test dataset exploration
python scripts/explore_datasets.py --list

# Download a small dataset
python scripts/explore_datasets.py --dataset trec --download
```

---

## Quick Start Commands

### 1. Download Teacher Model:
```bash
python scripts/download_deepseek_model.py
```

### 2. Download Dataset:
```bash
python scripts/explore_datasets.py --dataset dbpedia --validation-split 0.2
```

### 3. Train Student with Knowledge Distillation:
```bash
python scripts/train_soft_distillation.py \
  --dataset dbpedia \
  --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-model Qwen/Qwen2.5-0.5B \
  --epochs 3 \
  --batch-size 4 \
  --max-length 128
```

### 4. Quick Test (Small Subset):
```bash
python scripts/train_soft_distillation.py \
  --dataset dbpedia \
  --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-model Qwen/Qwen2.5-0.5B \
  --max-train-samples 100 \
  --max-val-samples 100 \
  --batch-size 2 \
  --epochs 2
```

---

## Environment Variables

Optional environment variables you can set:

```bash
# HuggingFace token (for private models)
export HF_TOKEN=your_token_here

# Cache directories
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export TORCH_HOME=~/.cache/torch

# WandB (for experiment tracking)
export WANDB_API_KEY=your_wandb_key
```

---

## Troubleshooting

### NumPy compatibility error:
```bash
pip install "numpy<2.0"
```

### CUDA out of memory:
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing

### Slow downloads:
```bash
# Set HuggingFace mirror (China users)
export HF_ENDPOINT=https://hf-mirror.com
```

---

## System Requirements

### Minimum:
- Python 3.10+
- 16GB RAM
- 50GB disk space

### Recommended:
- NVIDIA GPU with 8GB+ VRAM (for training)
- 32GB+ RAM
- 100GB+ SSD

### For Apple Silicon:
- M1/M2/M3/M4 chip
- 16GB+ unified memory
- PyTorch will use MPS backend automatically
