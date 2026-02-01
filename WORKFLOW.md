# Knowledge Distillation Workflow

## Overview

This project implements **online knowledge distillation** where teacher and student models run together during training:

```python
for epoch in epochs:
    for batch in data:
        # Teacher generates soft targets (frozen, eval mode)
        with torch.no_grad():
            teacher_logits = teacher(batch)
        
        # Student learns from both teacher and true labels
        student_logits = student(batch)
        loss = alpha × KL(student || teacher) + (1-alpha) × CE(student, labels)
        
        # Only student parameters updated
        loss.backward()
        optimizer.step()
```

## Why This Approach?

**Benefits:**
- ✅ **Simple workflow**: One command trains the student
- ✅ **Direct learning**: Student learns from teacher's current knowledge
- ✅ **No preprocessing**: No need to generate and store logits
- ✅ **Dynamic**: Teacher provides fresh soft targets each epoch

**Considerations:**
- Both models loaded in memory (use smaller models or reduce batch size)
- Teacher forward pass every batch (minimal overhead since frozen)
- Gradient checkpointing reduces memory usage

## Step-by-Step Workflow

### Step 1: Download Teacher Model

```bash
python scripts/download_deepseek_model.py
```

**Downloads:** DeepSeek-R1-Distill-Qwen-1.5B to `models/`

### Step 2: Download Dataset

```bash
python scripts/explore_datasets.py \
  --dataset dbpedia \
  --validation-split 0.2
```

**Downloads:** DBpedia to `data/dbpedia/` with train/validation/test splits

### Step 3: Train Student with Knowledge Distillation

```bash
python scripts/train_soft_distillation.py \
  --dataset dbpedia \
  --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-model Qwen/Qwen2.5-0.5B \
  --epochs 3 \
  --batch-size 4 \
  --max-length 128 \
  --temperature 3.0 \
  --alpha 0.7
```

**What it does:**
1. Loads teacher (frozen, eval mode) and student (trainable)
2. For each batch:
   - Teacher generates soft probability distribution
   - Student learns to match teacher's distribution (KL divergence)
   - Student also learns from true labels (cross-entropy)
3. Saves best model based on validation accuracy

**Output:**
```
outputs/student_soft_distill/dbpedia_T3.0_A0.7_*/
├── best_model/       # Best checkpoint (highest val accuracy)
├── final_model/      # Final checkpoint
├── history.json      # Training metrics per epoch
└── config.json       # Training configuration
```

### Step 4: Evaluate

```bash
python scripts/compare_models.py \
  --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-model outputs/student_soft_distill/.../best_model \
  --dataset dbpedia
```

**Output:**
- `comparison_results.json`: Detailed metrics
- `teacher_student_comparison.png`: Visualization

## Key Parameters

### Temperature (`--temperature`)
Controls how "soft" the probability distributions are:
- **Low (1.0-2.0)**: Sharp, confident predictions → Less knowledge transfer
- **Medium (2.0-4.0)**: Balanced → Good default
- **High (4.0+)**: Very smooth distributions → More knowledge transfer

Formula: `softmax(logits / T)`

### Alpha (`--alpha`)
Balances distillation vs supervised learning:
- **0.0**: Pure supervised (only CE loss) → No distillation
- **0.5**: Equal weight → Balanced
- **0.7**: Favor distillation (default) → More teacher influence
- **1.0**: Pure distillation (only KL loss) → Maximum teacher mimicking

Formula: `α * KL_loss + (1-α) * CE_loss`

## Quick Testing

Before full training, test with a small subset:

```bash
python scripts/train_soft_distillation.py \
  --dataset dbpedia \
  --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-model Qwen/Qwen2.5-0.5B \
  --max-train-samples 100 \
  --max-val-samples 100 \
  --batch-size 2 \
  --max-length 128 \
  --epochs 2
```

**Why this works:**
- `--max-train-samples` uses stratified sampling (maintains class distribution)
- Small batch size and max length reduce memory usage
- Quick iteration to find best hyperparameters
- Scale up by increasing samples and batch size once satisfied

**Memory optimization tips:**
- Start with 50-100 samples on Apple MPS
- Use `--batch-size 1` if OOM errors occur
- Reduce `--max-length` to 64 or 128
- Try smaller student: `prajjwal1/bert-tiny` (~4M params)

## Experiment Tracking

All runs are timestamped and organized:

```
outputs/student_soft_distill/
├── dbpedia_teacher_logits_T3.0_A0.7_20260131_135045/
├── dbpedia_teacher_logits_T4.0_A0.5_20260131_140230/
└── dbpedia_teacher_logits_T2.0_A0.8_20260131_141015/
```

Each directory contains:
- `history.json`: Training curves (loss, accuracy per epoch)
- `best_model/`: Checkpoint with highest validation accuracy
- `final_model/`: Final epoch checkpoint

## Common Issues

### Issue: Out of memory during training
**Solution 1:** Reduce batch size and max length:
```bash
python scripts/train_soft_distillation.py \
  --batch-size 1 \
  --max-length 64 \
  --max-train-samples 50
```

**Solution 2:** Use a smaller student model:
```bash
python scripts/train_soft_distillation.py \
  --student-model prajjwal1/bert-tiny  # Only 4M parameters
```

**Solution 3:** Use gradient accumulation (future improvement):
- Allows larger effective batch sizes with smaller micro-batches

### Issue: Student not learning
**Solution:** Check alpha value (might be too high):
```bash
python scripts/train_soft_distillation.py --alpha 0.5  # More supervised learning
```

### Issue: Student matches teacher too closely
**Solution:** Reduce temperature or alpha:
```bash
python scripts/train_soft_distillation.py --temperature 2.0 --alpha 0.5
```

## Next Steps

1. **Download** teacher model and dataset
2. **Quick test** with `--max-train-samples 100 --batch-size 2`
3. **Experiment** with temperature (2.0-4.0) and alpha (0.5-0.8)
4. **Scale up** by increasing samples and batch size
5. **Full training** on complete dataset
6. **Compare** results with `compare_models.py`

## Advanced Optimization

For better memory efficiency:
- Enable gradient checkpointing (automatic in script)
- Use mixed precision training (CUDA only)
- Consider CPU training for very large models (slower but works)
- Try smaller student architectures first

Happy distilling! 🎓→👨‍🎓
