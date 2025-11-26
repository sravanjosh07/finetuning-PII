"""
Find Best Threshold Values and Generate Confusion Matrices

Loads saved predictions and tests different threshold values to find optimal F1 scores.
Creates:
  1. F1 vs threshold plot for all models
  2. Confusion matrix (plot + CSV) for each model at their best threshold
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.label_normalizer import normalizer
from utils.models_config import LABEL_CONSOLIDATION_MAP

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PREDS_DIR = str(SCRIPT_DIR.parent / 'preds_24_labels')  # Where predictions are saved
OUTPUT_PLOT = str(SCRIPT_DIR.parent / 'results_24_labels' / 'threshold_analysis.png')

# Threshold range to test
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
STEP = 0.05  # Test every 0.05 (0.0, 0.05, 0.10, ..., 0.95, 1.0)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def labels_match(label1, label2):
    """Check if two labels match (exact or hierarchical)"""
    return normalizer.labels_match(label1, label2)


def calculate_metrics_at_threshold(predictions, ground_truth, threshold):
    """
    Calculate F1, Precision, Recall at a specific threshold.

    Args:
        predictions: List of prediction dicts with 'text', 'normalized_label', 'score'
        ground_truth: List of ground truth dicts with 'entity', 'normalized_type'
        threshold: Confidence threshold to apply

    Returns:
        Dict with precision, recall, f1
    """
    # Filter predictions by threshold
    filtered_preds = set()
    for p in predictions:
        if p['score'] >= threshold:
            entity_text = p['text'].lower().strip()
            # Apply label consolidation mapping
            normalized_label = p['normalized_label']
            mapped_label = LABEL_CONSOLIDATION_MAP.get(normalized_label, normalized_label)
            filtered_preds.add((entity_text, mapped_label))

    # Convert ground truth to set
    gt_set = set()
    for gt in ground_truth:
        entity_text = gt['entity'].lower().strip()
        # Apply label consolidation mapping
        normalized_type = gt['normalized_type']
        mapped_label = LABEL_CONSOLIDATION_MAP.get(normalized_type, normalized_type)
        gt_set.add((entity_text, mapped_label))

    # Calculate matches using hierarchical matching
    matched_preds = set()
    matched_gts = set()

    for pred_text, pred_label in filtered_preds:
        for gt_text, gt_label in gt_set:
            if pred_text == gt_text and labels_match(pred_label, gt_label):
                matched_preds.add((pred_text, pred_label))
                matched_gts.add((gt_text, gt_label))
                break

    # Calculate metrics
    tp = len(matched_preds)
    fp = len(filtered_preds) - len(matched_preds)
    fn = len(gt_set) - len(matched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def load_predictions(pred_file):
    """Load predictions from JSON file"""
    with open(pred_file, 'r') as f:
        data = json.load(f)
    return data


def analyze_model_thresholds(pred_file, thresholds):
    """
    Analyze a model's performance across different thresholds.

    Args:
        pred_file: Path to predictions JSON file
        thresholds: List of threshold values to test

    Returns:
        Dict with threshold -> metrics mapping
    """
    print(f"  Loading predictions from {Path(pred_file).name}...")
    data = load_predictions(pred_file)
    model_name = data['model']
    predictions_data = data['predictions']

    print(f"  Testing {len(thresholds)} threshold values...")
    results = {}

    for threshold in thresholds:
        # Calculate metrics for all samples at this threshold
        all_tp, all_fp, all_fn = 0, 0, 0

        for sample in predictions_data:
            predictions = sample['predictions']
            ground_truth = sample['ground_truth']

            metrics = calculate_metrics_at_threshold(predictions, ground_truth, threshold)
            all_tp += metrics['tp']
            all_fp += metrics['fp']
            all_fn += metrics['fn']

        # Calculate overall metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Find best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
    best_f1 = results[best_threshold]['f1']

    print(f"  ✓ Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")

    return model_name, results, best_threshold, best_f1


def create_confusion_matrix(pred_file, threshold, f1_score, output_dir):
    """
    Create confusion matrix for a model at a specific threshold.

    Args:
        pred_file: Path to predictions JSON file
        threshold: Threshold to use for filtering predictions
        f1_score: F1 score achieved at this threshold
        output_dir: Directory to save confusion matrix plot and CSV

    Returns:
        Confusion matrix as numpy array
    """
    data = load_predictions(pred_file)
    model_name = data['model']
    predictions_data = data['predictions']

    # Collect all true and predicted labels
    true_labels = []
    pred_labels = []

    for sample in predictions_data:
        ground_truth = sample['ground_truth']
        predictions = sample['predictions']

        # Filter predictions by threshold and apply label mapping
        filtered_preds = {}
        for p in predictions:
            if p['score'] >= threshold:
                entity_text = p['text'].lower().strip()
                normalized_label = p['normalized_label']
                mapped_label = LABEL_CONSOLIDATION_MAP.get(normalized_label, normalized_label)
                # Keep highest score for each entity
                if entity_text not in filtered_preds or p['score'] > filtered_preds[entity_text]['score']:
                    filtered_preds[entity_text] = {'label': mapped_label, 'score': p['score']}

        # Ground truth dict with label mapping
        gt_dict = {}
        for ent in ground_truth:
            entity_text = ent['entity'].lower().strip()
            normalized_type = ent['normalized_type']
            mapped_label = LABEL_CONSOLIDATION_MAP.get(normalized_type, normalized_type)
            gt_dict[entity_text] = mapped_label

        # For each ground truth entity
        for entity_text, true_label in gt_dict.items():
            true_labels.append(true_label)
            if entity_text in filtered_preds:
                pred_labels.append(filtered_preds[entity_text]['label'])
            else:
                pred_labels.append('MISSED')

        # For false positives (predictions not in ground truth)
        for entity_text, pred_info in filtered_preds.items():
            if entity_text not in gt_dict:
                true_labels.append('NO_ENTITY')
                pred_labels.append(pred_info['label'])

    # Get all unique labels (sorted)
    all_labels = sorted(set(true_labels) - {'NO_ENTITY'} | set(pred_labels) - {'MISSED'})
    display_labels = all_labels + ['NO_ENTITY', 'MISSED']

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=display_labels)

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    csv_path = Path(output_dir) / f'confusion_matrix_{model_name.replace(" ", "_").replace("-", "_")}.csv'
    cm_df.to_csv(csv_path)
    print(f"  ✓ Saved confusion matrix CSV to {csv_path.name}")

    # Create and save plot
    fig, ax = plt.subplots(figsize=(18, 16))

    # Normalize by row for color (percentage), but show counts as text
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_percent = cm / row_sums * 100

    # Create heatmap
    sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels,
                annot_kws={'size': 8},
                vmin=0, vmax=100, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}\nBest Threshold: {threshold:.2f} | F1 Score: {f1_score:.3f}',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Save plot
    plot_path = Path(output_dir) / f'confusion_matrix_{model_name.replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrix plot to {plot_path.name}")

    return cm


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("THRESHOLD ANALYSIS - Finding Optimal Thresholds for Each Model")
    print("=" * 100)

    # Get all prediction files
    pred_files = sorted(Path(PREDS_DIR).glob('predictions_*.json'))

    if not pred_files:
        print(f"\n❌ No prediction files found in {PREDS_DIR}")
        print("   Run compare_all_models_unified.py first to generate predictions.")
        return

    print(f"\nFound {len(pred_files)} model prediction files")
    print(f"Testing thresholds from {MIN_THRESHOLD} to {MAX_THRESHOLD} in steps of {STEP}")

    # Generate threshold values to test
    thresholds = np.arange(MIN_THRESHOLD, MAX_THRESHOLD + STEP, STEP)
    thresholds = [round(t, 2) for t in thresholds]  # Round to 2 decimals

    print(f"\n{'=' * 100}")
    print("ANALYZING MODELS")
    print('=' * 100)

    # Analyze each model
    all_model_results = {}
    best_thresholds = {}

    for pred_file in pred_files:
        print(f"\n{pred_file.stem}:")
        model_name, results, best_threshold, best_f1 = analyze_model_thresholds(
            str(pred_file), thresholds
        )
        all_model_results[model_name] = results
        best_thresholds[model_name] = {
            'threshold': best_threshold,
            'f1': best_f1,
            'precision': results[best_threshold]['precision'],
            'recall': results[best_threshold]['recall']
        }

    # Create plot
    print(f"\n{'=' * 100}")
    print("CREATING PLOT")
    print('=' * 100)

    plt.figure(figsize=(12, 7))

    # Plot each model as a line
    for model_name, results in all_model_results.items():
        thresholds_list = sorted(results.keys())
        f1_scores = [results[t]['f1'] for t in thresholds_list]

        # Plot line
        plt.plot(thresholds_list, f1_scores, marker='o', markersize=3,
                label=model_name, linewidth=2)

        # Mark best threshold
        best_t = best_thresholds[model_name]['threshold']
        best_f1 = best_thresholds[model_name]['f1']
        plt.scatter([best_t], [best_f1], s=100, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=5)

    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Model Performance vs Threshold (24 Labels)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.xlim(MIN_THRESHOLD, MAX_THRESHOLD)
    plt.ylim(0, 1)

    # Save plot
    os.makedirs(Path(OUTPUT_PLOT).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {OUTPUT_PLOT}")

    # Display best thresholds summary
    print(f"\n{'=' * 100}")
    print("BEST THRESHOLDS SUMMARY")
    print('=' * 100)
    print(f"{'MODEL':<25} {'THRESHOLD':<12} {'F1':<12} {'PRECISION':<12} {'RECALL':<12}")
    print('=' * 100)

    # Sort by F1 score
    sorted_models = sorted(best_thresholds.items(),
                          key=lambda x: x[1]['f1'], reverse=True)

    for model_name, metrics in sorted_models:
        print(f"{model_name:<25} {metrics['threshold']:<12.2f} "
              f"{metrics['f1']:<12.3f} {metrics['precision']:<12.3f} "
              f"{metrics['recall']:<12.3f}")

    print('=' * 100)

    # Save results to JSON
    output_json = str(Path(OUTPUT_PLOT).parent / 'best_thresholds.json')
    with open(output_json, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\n✓ Saved best thresholds to {output_json}")

    # Generate confusion matrices at best thresholds
    print(f"\n{'=' * 100}")
    print("GENERATING CONFUSION MATRICES")
    print('=' * 100)

    output_dir = Path(OUTPUT_PLOT).parent
    for pred_file in pred_files:
        data = load_predictions(str(pred_file))
        model_name = data['model']
        best_threshold = best_thresholds[model_name]['threshold']
        best_f1 = best_thresholds[model_name]['f1']

        print(f"\n{model_name}:")
        print(f"  Creating confusion matrix at best threshold {best_threshold:.2f} (F1: {best_f1:.3f})...")
        create_confusion_matrix(str(pred_file), best_threshold, best_f1, str(output_dir))

    print(f"\n{'=' * 100}")
    print("THRESHOLD ANALYSIS COMPLETE ✓")
    print('=' * 100)
    print(f"\nOutputs saved in: {output_dir}/")
    print("  - threshold_analysis.png (F1 vs threshold plot)")
    print("  - best_thresholds.json (best threshold for each model)")
    print("  - confusion_matrix_*.png (confusion matrix plots)")
    print("  - confusion_matrix_*.csv (confusion matrix data)")


if __name__ == "__main__":
    main()
