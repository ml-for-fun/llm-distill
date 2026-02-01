#!/usr/bin/env python3
"""
Generate teacher model's logits for training/validation data.
This allows efficient distillation without loading teacher during training.

Usage:
    python scripts/generate_teacher_logits.py --dataset dbpedia --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, Dataset
from tqdm import tqdm


def stratified_sample(dataset, num_samples, label_col="label", seed=42):
    """
    Sample dataset while maintaining class distribution.
    
    Args:
        dataset: HuggingFace Dataset
        num_samples: Total number of samples to select
        label_col: Name of label column
        seed: Random seed
    
    Returns:
        Sampled dataset with same class proportions
    """
    np.random.seed(seed)
    
    # Get all labels
    all_labels = dataset[label_col]
    unique_labels = np.unique(all_labels)
    
    # Count samples per class
    label_counts = defaultdict(int)
    for label in all_labels:
        label_counts[label] += 1
    
    # Calculate proportional samples per class
    total_samples = len(dataset)
    samples_per_class = {}
    for label in unique_labels:
        proportion = label_counts[label] / total_samples
        samples_per_class[label] = max(1, int(num_samples * proportion))
    
    # Adjust to match exact num_samples
    while sum(samples_per_class.values()) < num_samples:
        # Add to largest class
        max_class = max(samples_per_class.keys(), key=lambda k: label_counts[k])
        samples_per_class[max_class] += 1
    
    while sum(samples_per_class.values()) > num_samples:
        # Remove from largest class
        max_class = max(samples_per_class.keys(), key=lambda k: samples_per_class[k])
        samples_per_class[max_class] -= 1
    
    # Sample indices for each class
    selected_indices = []
    for label in unique_labels:
        # Get all indices for this class
        class_indices = [i for i, l in enumerate(all_labels) if l == label]
        # Sample required number
        n_samples = samples_per_class[label]
        sampled = np.random.choice(class_indices, size=min(n_samples, len(class_indices)), replace=False)
        selected_indices.extend(sampled.tolist())
    
    # Return sampled dataset
    return dataset.select(selected_indices)


def setup_device():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    return device


def collate_fn(batch, tokenizer, max_length=512):
    """Collate function for DataLoader."""
    texts = [item["content"] for item in batch]
    labels = [item["label"] for item in batch]
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts  # Keep original text
    }


@torch.no_grad()
def generate_logits(model, dataloader, device, split_name):
    """Generate teacher logits for entire dataset."""
    model.eval()
    
    all_logits = []
    all_labels = []
    all_texts = []
    
    print(f"\n🔮 Generating teacher logits for {split_name} set...")
    pbar = tqdm(dataloader, desc=f"Processing {split_name}")
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        texts = batch["texts"]
        
        # Get teacher predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Store logits (before softmax)
        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)
        all_labels.extend(labels.numpy())
        all_texts.extend(texts)
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)
    
    print(f"   ✅ Generated {len(all_logits):,} samples")
    print(f"   Shape: {all_logits.shape}")
    
    return all_texts, all_labels, all_logits


def main():
    parser = argparse.ArgumentParser(description="Generate teacher logits for distillation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., dbpedia)")
    parser.add_argument("--teacher-model", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples (stratified)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples (stratified)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data/{dataset}_teacher_logits)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path("data") / args.dataset
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create meaningful name with sample counts if limited
        suffix = "teacher_logits"
        if args.max_train_samples or args.max_val_samples:
            parts = []
            if args.max_train_samples:
                parts.append(f"train{args.max_train_samples}")
            if args.max_val_samples:
                parts.append(f"val{args.max_val_samples}")
            suffix = f"teacher_logits_{'_'.join(parts)}"
        output_dir = Path("data") / f"{args.dataset}_{suffix}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎓 Teacher Logits Generation")
    print(f"   Dataset: {args.dataset}")
    print(f"   Teacher: {args.teacher_model}")
    print(f"   Output: {output_dir}")
    
    # Setup device
    device = setup_device()
    
    # Load teacher model and tokenizer
    print(f"\n📥 Loading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)
    
    # Set padding token in model config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(device)
    
    # Force float32 on MPS
    if device.type == "mps":
        model = model.to(torch.float32)
        print("   Using float32 for MPS compatibility")
    
    print(f"   ✅ Teacher loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")
    
    # Get number of classes from model
    num_classes = model.config.num_labels
    print(f"   Number of classes: {num_classes}")
    
    # Process each split
    for split in ["train", "validation"]:
        split_path = data_dir / split
        
        if not split_path.exists():
            print(f"\n⚠️  Skipping {split} (not found)")
            continue
        
        # Load dataset
        dataset = load_from_disk(str(split_path))
        original_size = len(dataset)
        print(f"\n📊 Processing {split} set: {original_size:,} samples")
        
        # Apply stratified sampling if max samples specified
        if split == "train" and args.max_train_samples and args.max_train_samples < len(dataset):
            print(f"   🎯 Stratified sampling: {args.max_train_samples:,} samples")
            dataset = stratified_sample(dataset, args.max_train_samples, "label", args.seed)
            print(f"   ✓ Sampled {len(dataset):,} / {original_size:,} (maintaining class distribution)")
        elif split == "validation" and args.max_val_samples and args.max_val_samples < len(dataset):
            print(f"   🎯 Stratified sampling: {args.max_val_samples:,} samples")
            dataset = stratified_sample(dataset, args.max_val_samples, "label", args.seed)
            print(f"   ✓ Sampled {len(dataset):,} / {original_size:,} (maintaining class distribution)")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length)
        )
        
        # Generate logits
        texts, labels, logits = generate_logits(model, dataloader, device, split)
        
        # Create augmented dataset
        augmented_data = {
            "content": texts,
            "label": labels.tolist(),
            "teacher_logits": logits.tolist()
        }
        
        augmented_dataset = Dataset.from_dict(augmented_data)
        
        # Save to disk
        output_split_dir = output_dir / split
        augmented_dataset.save_to_disk(str(output_split_dir))
        print(f"   💾 Saved to {output_split_dir}")
    
    # Save metadatanum_classes,  # Use from model config, not logits shape
    metadata = {
        "dataset": args.dataset,
        "teacher_model": args.teacher_model,
        "num_classes": int(logits.shape[1]),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "seed": args.seed
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Teacher logits generation complete!")
    print(f"📁 Output saved to: {output_dir}")
    print(f"\n💡 Now train student with:")
    dataset_name = output_dir.name
    print(f"   python scripts/train_soft_distillation.py \\")
    print(f"     --dataset {dataset_name} \\")
    print(f"     --use-cached-logits \\")
    print(f"     --epochs 3 --batch-size 8")


if __name__ == "__main__":
    main()
