#!/usr/bin/env python3
"""
Train student model using Knowledge Distillation (KD) with soft targets.
Uses KL divergence to match student's output distribution to teacher's.

Usage:
    python scripts/train_soft_distillation.py --dataset dbpedia --epochs 3 --batch-size 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def stratified_sample(dataset, num_samples, label_col="label", seed=42):
    """
    Perform stratified sampling to maintain class distribution.
    
    Args:
        dataset: HuggingFace dataset
        num_samples: Number of samples to select
        label_col: Column name containing labels
        seed: Random seed for reproducibility
    
    Returns:
        Sampled dataset with proportional class distribution
    """
    if num_samples >= len(dataset):
        return dataset
    
    # Get all labels
    labels = dataset[label_col]
    unique_labels = list(set(labels))
    
    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    # Calculate samples per class (proportional)
    np.random.seed(seed)
    total_samples = len(dataset)
    selected_indices = []
    
    for label in unique_labels:
        label_indices = label_to_indices[label]
        # Proportional sampling
        n_label_samples = int(len(label_indices) * num_samples / total_samples)
        # Ensure at least 1 sample per class if possible
        n_label_samples = max(1, n_label_samples)
        
        # Random sample from this class
        if n_label_samples >= len(label_indices):
            selected_indices.extend(label_indices)
        else:
            sampled = np.random.choice(label_indices, size=n_label_samples, replace=False)
            selected_indices.extend(sampled.tolist())
    
    # If we have too many samples, trim randomly
    if len(selected_indices) > num_samples:
        selected_indices = np.random.choice(selected_indices, size=num_samples, replace=False).tolist()
    
    # Sort for consistent ordering
    selected_indices.sort()
    
    return dataset.select(selected_indices)


def setup_device():
    """Setup training device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float32  # Can use float16/bfloat16 on CUDA
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS has issues with float16
        print("✅ Using Apple MPS")
        print("⚠️  Memory tip: reduce --batch-size and --max-length if OOM")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️  Using CPU (training will be slow)")
    return device, dtype


def load_dataset_splits(dataset_path: str, max_train_samples=None, max_val_samples=None, label_col="label", seed=42):
    """Load train/val/test splits from disk with optional sampling."""
    dataset_path = Path(dataset_path)
    
    splits = {}
    for split in ["train", "validation"]:
        split_path = dataset_path / split
        if split_path.exists():
            # Load dataset
            dataset = load_from_disk(str(split_path))
            original_size = len(dataset)
            
            # Apply stratified sampling immediately to save memory
            if split == "train" and max_train_samples and max_train_samples < original_size:
                print(f"📂 Loading {split}: {original_size:,} samples")
                print(f"   🎯 Stratified sampling: {max_train_samples:,} samples")
                dataset = stratified_sample(dataset, max_train_samples, label_col, seed)
                print(f"   ✓ Sampled {len(dataset):,} samples (maintaining class distribution)")
            elif split == "validation" and max_val_samples and max_val_samples < original_size:
                print(f"📂 Loading {split}: {original_size:,} samples")
                print(f"   🎯 Stratified sampling: {max_val_samples:,} samples")
                dataset = stratified_sample(dataset, max_val_samples, label_col, seed)
                print(f"   ✓ Sampled {len(dataset):,} samples (maintaining class distribution)")
            else:
                print(f"📂 Loaded {split}: {len(dataset):,} samples")
            
            splits[split] = dataset
        else:
            print(f"⚠️  {split} split not found at {split_path}")
    
    if not splits:
        raise ValueError(f"No dataset splits found at {dataset_path}")
    
    return splits


def collate_fn(batch, tokenizer, max_length=256, use_cached_logits=False):
    """Collate batch into tensors."""
    texts = [item["content"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    # Tokenize
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    result = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels
    }
    
    # Add cached teacher logits if available
    if use_cached_logits:
        teacher_logits = [item["teacher_logits"] for item in batch]
        result["teacher_logits"] = torch.tensor(teacher_logits, dtype=torch.float32)
    
    return result


def prepare_dataset(dataset, tokenizer, text_col: str, label_col: str, max_length: int = 256):
    """Tokenize and prepare dataset for training."""
    
    def tokenize_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        # Add labels
        tokenized["labels"] = examples[label_col]
        return tokenized
    
    # Tokenize dataset
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Set format for PyTorch
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute distillation loss combining:
    - KL divergence between soft targets (student and teacher)
    - Cross-entropy with ground truth labels
    
    Args:
        student_logits: Student model outputs [batch_size, num_classes]
        teacher_logits: Teacher model outputs [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        temperature: Temperature for softening distributions
        alpha: Weight for KL loss (1-alpha for CE loss)
    
    Returns:
        Combined loss
    """
    # Soft targets from teacher (with temperature)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss (scaled by temperature^2)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets loss (standard cross-entropy)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    
    return loss, kl_loss.item(), ce_loss.item()


def compute_metrics(predictions, labels):
    """Compute accuracy and per-class metrics."""
    preds = predictions.argmax(axis=-1)
    acc = (preds == labels).mean()
    
    # Per-class accuracy
    num_classes = predictions.shape[-1]
    class_acc = []
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc.append((preds[mask] == labels[mask]).mean())
        else:
            class_acc.append(0.0)
    
    return {
        "accuracy": float(acc),
        "mean_class_accuracy": float(np.mean(class_acc))
    }


def train_epoch(student_model, teacher_model, dataloader, optimizer, scheduler, device, epoch, temperature, alpha, use_cached_logits=False):
    """Train for one epoch with knowledge distillation."""
    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()  # Teacher is always in eval mode
    
    total_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Get teacher logits (either from cache or live inference)
        if use_cached_logits:
            teacher_logits = batch["teacher_logits"].to(device)
        else:
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        loss, kl_loss, ce_loss = distillation_loss(
            student_logits, teacher_logits, labels, temperature, alpha
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        total_kl_loss += kl_loss
        total_ce_loss += ce_loss
        all_preds.append(student_logits.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        # Clear cache every few batches on MPS to prevent memory buildup
        if device.type == "mps" and batch_idx % 10 == 0:
            torch.mps.empty_cache()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "kl": f"{kl_loss:.4f}",
            "ce": f"{ce_loss:.4f}"
        })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, avg_kl_loss, avg_ce_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, device, split_name="validation"):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        total_loss += outputs.loss.item()
        all_preds.append(outputs.logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train student with soft distillation (KD)")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="dbpedia", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--text-col", type=str, default="content", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--num-classes", type=int, default=14, help="Number of classes")
    
    # Model arguments
    parser.add_argument("--teacher-model", type=str, 
                        default="models/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Teacher model path or name (not used if --use-cached-logits)")
    parser.add_argument("--student-model", type=str, 
                        default="Qwen/Qwen2.5-0.5B",
                        help="Student model name")
    parser.add_argument("--use-cached-logits", action="store_true",
                        help="Use pre-computed teacher logits (much faster, less memory)")
    parser.add_argument("--max-length", type=int, default=64, help="Max sequence length (reduce if OOM)")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=3.0, 
                        help="Temperature for softening logits (higher=softer)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for KL loss (1-alpha for CE loss)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples for quick testing (default: use all)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit validation samples for faster evaluation (default: use all)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs/student_soft_distill", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device, dtype = setup_device()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_T{args.temperature}_A{args.alpha}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset with immediate sampling to save memory
    print(f"\n📊 Loading dataset: {args.dataset}")
    dataset_path = Path(args.data_dir) / args.dataset
    splits = load_dataset_splits(
        dataset_path,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        label_col=args.label_col,
        seed=args.seed
    )
    
    if "train" not in splits or "validation" not in splits:
        raise ValueError("Need both train and validation splits")
    
    # Infer num_classes from dataset if using cached logits
    if args.use_cached_logits:
        # Try to read from metadata first
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                args.num_classes = metadata.get("num_classes", args.num_classes)
                print(f"   ✓ Inferred {args.num_classes} classes from metadata")
        else:
            # Infer from teacher_logits shape in first sample
            sample = splits["train"][0]
            if "teacher_logits" in sample:
                args.num_classes = len(sample["teacher_logits"])
                print(f"   ✓ Inferred {args.num_classes} classes from teacher_logits shape")
    
    # Load tokenizer (use student's tokenizer for both)
    print(f"\n🔧 Loading tokenizer from student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher model (only if not using cached logits)
    teacher_model = None
    
    if not args.use_cached_logits:
        print(f"\n👨‍🏫 Loading teacher model: {args.teacher_model}")
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_model,
            num_labels=args.num_classes,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        
        # Set pad token for teacher
        if teacher_model.config.pad_token_id is None:
            teacher_model.config.pad_token_id = tokenizer.pad_token_id
        
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # Teacher is frozen
        
        # Freeze teacher parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"   Parameters: {teacher_params:,}")
        print(f"   Status: Frozen (inference only)")
    else:
        print(f"\n⚡ Using cached teacher logits (no teacher model loaded)")
    
    # Load student model
    print(f"\n👨‍🎓 Loading student model: {args.student_model}")
    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=args.num_classes,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # Set pad token for student
    if student_model.config.pad_token_id is None:
        student_model.config.pad_token_id = tokenizer.pad_token_id
    
    student_model = student_model.to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(student_model, 'gradient_checkpointing_enable'):
        student_model.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing enabled (saves memory)")
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"   Parameters: {student_params:,}")
    print(f"   Trainable: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")
    if teacher_model is not None:
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"   Compression ratio: {teacher_params/student_params:.2f}x smaller")
    
    # Prepare datasets (sampling already done in load_dataset_splits)
    print(f"\n🔧 Preparing datasets...")
    train_dataset = splits["train"]
    val_dataset = splits["validation"]
    
    # Create dataloaders with dynamic collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, args.use_cached_logits)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, args.use_cached_logits)
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n🏋️  Training configuration:")
    print(f"   Distillation: KL divergence (soft targets)")
    print(f"   Mode: {'Cached logits (offline)' if args.use_cached_logits else 'Online (teacher + student)'}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Alpha (KL weight): {args.alpha}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    print(f"   Batches per epoch: {len(train_loader):,}")
    
    # Training loop
    print(f"\n🚀 Starting distillation training...\n")
    best_val_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train with distillation
        train_loss, train_kl, train_ce, train_metrics = train_epoch(
            student_model, teacher_model, train_loader, optimizer, scheduler, 
            device, epoch, args.temperature, args.alpha, args.use_cached_logits
        )
        
        # Validate
        val_loss, val_metrics = evaluate(student_model, val_loader, device, "validation")
        
        # Log results
        print(f"\n📊 Epoch {epoch}/{args.epochs} Results:")
        print(f"   Train Loss: {train_loss:.4f} (KL: {train_kl:.4f}, CE: {train_ce:.4f}) | Acc: {train_metrics['accuracy']:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_kl_loss": train_kl,
            "train_ce_loss": train_ce,
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"]
        })
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            print(f"   💾 New best validation accuracy! Saving model...")
            student_model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
        
        print()
    
    # Save final model
    print("💾 Saving final model...")
    student_model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Distillation training complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    # Print instructions for testing
    print(f"\n💡 To evaluate on test set and compare with teacher, run:")
    print(f"   python scripts/compare_models.py \\")
    if args.use_cached_logits:
        # Extract base dataset name (remove _teacher_logits suffix)
        base_dataset = args.dataset.replace("_teacher_logits", "").replace("_train", "").split("_val")[0]
        print(f"     --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \\")
        print(f"     --student-model {output_dir / 'best_model'} \\")
        print(f"     --dataset {base_dataset}")
    else:
        print(f"     --teacher-model {args.teacher_model} \\")
        print(f"     --student-model {output_dir / 'best_model'} \\")
        print(f"     --dataset {args.dataset}")


if __name__ == "__main__":
    main()
