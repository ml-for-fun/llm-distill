#!/usr/bin/env python3
"""
Compare teacher and student models on test set.
Generates detailed metrics and visualization.

Usage:
    python scripts/compare_models.py \
        --teacher-model models/DeepSeek-R1-Distill-Qwen-1.5B \
        --student-model outputs/student_soft_distill/.../best_model \
        --dataset dbpedia
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
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
    """Setup evaluation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float32
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
        print("✅ Using Apple MPS (using float32)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️  Using CPU")
    return device, dtype


def prepare_dataset(dataset, tokenizer, text_col: str, label_col: str, max_length: int = 256):
    """Tokenize and prepare dataset."""
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        tokenized["labels"] = examples[label_col]
        return tokenized
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized


@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name="model"):
    """
    Evaluate model with detailed metrics.
    Returns accuracy, precision, recall, F1.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating {model_name}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        all_preds.append(outputs.logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Get predicted classes
    pred_classes = all_preds.argmax(axis=-1)
    
    # Compute detailed metrics
    accuracy = accuracy_score(all_labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_classes, average='macro', zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def create_comparison_chart(teacher_metrics, student_metrics, output_path, dataset_name):
    """Generate bar chart comparing teacher and student."""
    metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
    teacher_vals = [teacher_metrics["accuracy"], teacher_metrics["precision"], 
                   teacher_metrics["recall"], teacher_metrics["f1"]]
    student_vals = [student_metrics["accuracy"], student_metrics["precision"], 
                   student_metrics["recall"], student_metrics["f1"]]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, student_vals, width, label='Student', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Teacher vs Student Performance on {dataset_name.upper()} Test Set', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare teacher and student models on test set")
    
    # Model arguments
    parser.add_argument("--teacher-model", type=str, required=True,
                        help="Path to teacher model")
    parser.add_argument("--student-model", type=str, required=True,
                        help="Path to student model (best_model directory)")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="dbpedia", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--text-col", type=str, default="content", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--num-classes", type=int, default=14, help="Number of classes")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max test samples to evaluate (default: use all)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as student model)")
    
    args = parser.parse_args()
    
    # Setup device
    device, dtype = setup_device()
    
    # Load test dataset
    print(f"\n📊 Loading test dataset: {args.dataset}")
    dataset_path = Path(args.data_dir) / args.dataset / "test"
    
    if not dataset_path.exists():
        raise ValueError(f"Test dataset not found at {dataset_path}")
    
    test_data = load_from_disk(str(dataset_path))
    print(f"   Test samples: {len(test_data):,}")
    
    # Apply stratified sampling if max_samples specified
    if args.max_samples and args.max_samples < len(test_data):
        print(f"   🎯 Stratified sampling: {args.max_samples:,} samples")
        test_data = stratified_sample(test_data, args.max_samples, args.label_col, seed=42)
        print(f"   ✓ Sampled {len(test_data):,} samples (maintaining class distribution)")
    
    # Load tokenizer from student model
    print(f"\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare test dataset
    print(f"\n🔧 Preparing test dataset...")
    test_dataset = prepare_dataset(test_data, tokenizer, args.text_col, args.label_col, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load teacher model
    print(f"\n👨‍🏫 Loading teacher model: {args.teacher_model}")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model,
        num_labels=args.num_classes,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    if teacher_model.config.pad_token_id is None:
        teacher_model.config.pad_token_id = tokenizer.pad_token_id
    
    teacher_model = teacher_model.to(device)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"   Parameters: {teacher_params:,}")
    
    # Load student model
    print(f"\n👨‍🎓 Loading student model: {args.student_model}")
    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    if student_model.config.pad_token_id is None:
        student_model.config.pad_token_id = tokenizer.pad_token_id
    
    student_model = student_model.to(device)
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"   Parameters: {student_params:,}")
    print(f"   Compression ratio: {teacher_params/student_params:.2f}x smaller")
    
    # Evaluate both models
    print(f"\n🧪 Evaluating models on test set...\n")
    
    teacher_metrics = evaluate_model(teacher_model, test_loader, device, "Teacher")
    student_metrics = evaluate_model(student_model, test_loader, device, "Student")
    
    # Print comparison
    print(f"\n📊 Test Results Comparison:")
    print(f"\n{'Metric':<15} {'Teacher':<12} {'Student':<12} {'Difference':<12}")
    print("-" * 55)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        teacher_val = teacher_metrics[metric]
        student_val = student_metrics[metric]
        diff = student_val - teacher_val
        diff_str = f"{diff:+.4f}"
        print(f"{metric.capitalize():<15} {teacher_val:<12.4f} {student_val:<12.4f} {diff_str:<12}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save in student model's parent directory
        output_dir = Path(args.student_model).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison chart
    print(f"\n📈 Generating comparison chart...")
    chart_path = output_dir / "teacher_student_comparison.png"
    create_comparison_chart(teacher_metrics, student_metrics, chart_path, args.dataset)
    print(f"   ✅ Chart saved to: {chart_path}")
    
    # Save detailed results
    results = {
        "teacher": teacher_metrics,
        "student": student_metrics,
        "difference": {
            "accuracy": student_metrics["accuracy"] - teacher_metrics["accuracy"],
            "precision": student_metrics["precision"] - teacher_metrics["precision"],
            "recall": student_metrics["recall"] - teacher_metrics["recall"],
            "f1": student_metrics["f1"] - teacher_metrics["f1"]
        },
        "model_info": {
            "teacher_path": args.teacher_model,
            "student_path": args.student_model,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": teacher_params / student_params
        },
        "dataset": args.dataset,
        "test_samples": len(test_data)
    }
    
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Results saved to: {results_path}")
    
    print(f"\n✅ Comparison complete!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
