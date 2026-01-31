#!/usr/bin/env python3
"""
Explore and download multiclass classification datasets for LLM distillation.

Recommended datasets for distillation evaluation:
1. AG News (4 classes) - News categorization
2. DBpedia (14 classes) - Topic classification  
3. TREC (6 classes) - Question type classification
4. Emotion (6 classes) - Emotion detection
5. Banking77 (77 classes) - Banking intent classification
6. Yahoo Answers Topics (10 classes) - Question topic classification

Usage:
    python scripts/explore_datasets.py --dataset ag_news --preview
    python scripts/explore_datasets.py --dataset trec --download
"""

import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed. Install with:")
    print("  pip install datasets")
    sys.exit(1)


DATASET_CONFIGS = {
    "ag_news": {
        "name": "fancyzhx/ag_news",
        "config": None,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 4,
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "description": "News articles categorization - 4 balanced classes, 120k training samples"
    },
    "trec": {
        "name": "trec",
        "config": None,
        "text_col": "text",
        "label_col": "coarse_label",
        "num_classes": 6,
        "classes": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "description": "Question type classification - 6 classes, 5.5k training samples"
    },
    "emotion": {
        "name": "dair-ai/emotion",
        "config": None,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 6,
        "classes": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "description": "Emotion detection from text - 6 classes, 16k training samples"
    },
    "dbpedia": {
        "name": "fancyzhx/dbpedia_14",
        "config": None,
        "text_col": "content",
        "label_col": "label",
        "num_classes": 14,
        "classes": [
            "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
            "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
            "Plant", "Album", "Film", "WrittenWork"
        ],
        "description": "DBpedia ontology classification - 14 classes, 560k training samples"
    },
    "banking77": {
        "name": "PolyAI/banking77",
        "config": None,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 77,
        "classes": ["activate_my_card", "age_limit", "apple_pay_or_google_pay", "..."],
        "description": "Banking customer intent - 77 fine-grained classes, 10k training samples"
    },
    "yahoo_answers": {
        "name": "yahoo_answers_topics",
        "config": None,
        "text_col": "best_answer",
        "label_col": "topic",
        "num_classes": 10,
        "classes": [
            "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
            "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government"
        ],
        "description": "Yahoo Answers topic classification - 10 classes, 1.4M training samples"
    }
}


def split_train_validation(dataset, val_ratio: float = 0.1, seed: int = 42):
    """Split training data into train and validation sets."""
    if 'train' not in dataset:
        print("⚠️  No 'train' split found to split")
        return dataset
    
    print(f"\n✂️  Splitting train into train/validation ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)...")
    split_dataset = dataset['train'].train_test_split(test_size=val_ratio, seed=seed)
    
    # Replace train with new split, rename test to validation
    dataset['train'] = split_dataset['train']
    dataset['validation'] = split_dataset['test']
    
    print(f"   ✓ New train size: {len(dataset['train']):,}")
    print(f"   ✓ New validation size: {len(dataset['validation']):,}")
    
    return dataset


def preview_dataset(dataset_key: str, num_samples: int = 5, val_ratio: float = 0.1):
    """Preview a dataset with examples and statistics."""
    if dataset_key not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available: {', '.join(DATASET_CONFIGS.keys())}")
        return
    
    config = DATASET_CONFIGS[dataset_key]
    print(f"\n📊 Dataset: {config['name']}")
    print(f"📝 Description: {config['description']}")
    print(f"🏷️  Classes ({config['num_classes']}): {', '.join(config['classes'][:5])}{'...' if len(config['classes']) > 5 else ''}")
    
    print(f"\n⏬ Loading dataset...")
    try:
        if config['config']:
            dataset = load_dataset(config['name'], config['config'])
        else:
            dataset = load_dataset(config['name'])
        
        # Auto-split if no validation set exists
        if 'validation' not in dataset and 'train' in dataset:
            dataset = split_train_validation(dataset, val_ratio=val_ratio)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"\n📈 Splits available: {list(dataset.keys())}")
        
        if 'train' in dataset:
            train_size = len(dataset['train'])
            print(f"   - Train: {train_size:,} samples")
        if 'test' in dataset:
            test_size = len(dataset['test'])
            print(f"   - Test: {test_size:,} samples")
        if 'validation' in dataset:
            val_size = len(dataset['validation'])
            print(f"   - Validation: {val_size:,} samples")
        
        # Show example
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        print(f"\n🔍 Sample from '{split}' split:")
        print("-" * 80)
        
        for i in range(min(num_samples, len(dataset[split]))):
            example = dataset[split][i]
            text = example[config['text_col']]
            label = example[config['label_col']]
            
            # Truncate long text
            if len(text) > 200:
                text = text[:200] + "..."
            
            # Get label name if available
            if hasattr(dataset[split].features[config['label_col']], 'names'):
                label_names = dataset[split].features[config['label_col']].names
                label_str = f"{label} ({label_names[label]})"
            else:
                label_str = str(label)
            
            print(f"\nExample {i+1}:")
            print(f"  Label: {label_str}")
            print(f"  Text: {text}")
        
        print("-" * 80)
        
        # Class distribution
        if 'train' in dataset:
            print(f"\n📊 Class distribution (train split):")
            labels = dataset['train'][config['label_col']]
            from collections import Counter
            label_counts = Counter(labels)
            
            for label_id in sorted(label_counts.keys())[:10]:  # Show first 10
                count = label_counts[label_id]
                percentage = 100 * count / len(labels)
                if hasattr(dataset['train'].features[config['label_col']], 'names'):
                    label_names = dataset['train'].features[config['label_col']].names
                    label_name = label_names[label_id]
                else:
                    label_name = str(label_id)
                print(f"   {label_name:30s}: {count:6d} ({percentage:5.2f}%)")
            
            if len(label_counts) > 10:
                print(f"   ... and {len(label_counts) - 10} more classes")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return


def download_dataset(dataset_key: str, output_dir: str = None, val_ratio: float = 0.1):
    """Download and cache a dataset."""
    if dataset_key not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available: {', '.join(DATASET_CONFIGS.keys())}")
        return
    
    config = DATASET_CONFIGS[dataset_key]
    
    # Default to local data directory if not specified
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Save to ./data/<dataset_name> by default
        script_dir = Path(__file__).parent
        output_path = script_dir.parent / "data" / dataset_key
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_path.absolute()}")
    
    print(f"⏬ Downloading {config['name']}...")
    
    try:
        # Load dataset (HuggingFace will cache in default location)
        if config['config']:
            dataset = load_dataset(config['name'], config['config'])
        else:
            dataset = load_dataset(config['name'])
        
        # Auto-split if no validation set exists
        if 'validation' not in dataset and 'train' in dataset:
            dataset = split_train_validation(dataset, val_ratio=val_ratio)
        
        print(f"✅ Dataset downloaded!")
        print(f"📊 Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        
        # Save each split to clean organized folders
        print(f"\n💾 Saving splits to clean structure...")
        for split_name in dataset.keys():
            split_dir = output_path / split_name
            dataset[split_name].save_to_disk(str(split_dir))
            num_samples = len(dataset[split_name])
            print(f"   ✓ {split_name:12s} → {split_dir.name}/ ({num_samples:,} samples)")
        
        print(f"\n📁 Dataset structure:")
        print(f"   {output_path.name}/")
        for split_name in sorted(dataset.keys()):
            print(f"   ├── {split_name}/")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def list_datasets():
    """List all available datasets."""
    print("\n📚 Available Multiclass Classification Datasets:\n")
    
    for i, (key, config) in enumerate(DATASET_CONFIGS.items(), 1):
        print(f"{i}. {key.upper()}")
        print(f"   Name: {config['name']}")
        print(f"   Classes: {config['num_classes']}")
        print(f"   Description: {config['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Explore multiclass datasets for LLM distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python scripts/explore_datasets.py --list
  
  # Preview AG News dataset
  python scripts/explore_datasets.py --dataset ag_news --preview
  
  # Download and cache TREC dataset
  python scripts/explore_datasets.py --dataset trec --download --output-dir ./data
  
  # Show more samples in preview
  python scripts/explore_datasets.py --dataset emotion --preview --num-samples 10
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to explore"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview dataset with sample examples"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and cache the dataset"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to show in preview (default: 5)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio if no validation set exists (default: 0.1 = 10%%)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded dataset (default: ./data/<dataset_name>)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.dataset:
        if args.preview:
            preview_dataset(args.dataset, args.num_samples, args.val_ratio)
        if args.download:
            download_dataset(args.dataset, args.output_dir, args.val_ratio)
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --list to see all available datasets")


if __name__ == "__main__":
    main()
