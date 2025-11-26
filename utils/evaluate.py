"""
Shared evaluation functions
Used by all evaluation scripts to avoid duplication

Refactored to use clean LabelNormalizer system for simplified label handling.
"""
from .label_normalizer import normalizer
from .post_processing import post_process_predictions, get_post_processing_stats
import time
import json
import os
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm


# For backwards compatibility, keep these functions but they now use the normalizer
def labels_match(label1: str, label2: str) -> bool:
    """
    Check if two labels match either exactly or within the same category family.
    
    Now delegates to the unified LabelNormalizer.

    Args:
        label1: First normalized label
        label2: Second normalized label

    Returns:
        True if labels match (exact or hierarchical), False otherwise
    """
    return normalizer.labels_match(label1, label2)


def evaluate_model(model, model_name, labels, test_data, threshold=0.3, verbose=True,
                   enable_post_processing=False, deduplicate=True, filter_code=True):
    """
    Evaluate a single model on test data.

    Now simplified: all normalization happens through LabelNormalizer.
    Works with both raw and normalized test data seamlessly.
    Returns both exact and hierarchical matching scores.

    Args:
        model: GLiNER model instance
        model_name: Name of the model (for model-specific label mapping)
        labels: List of labels to use for prediction
        test_data: List of test samples
        threshold: Confidence threshold
        verbose: Print progress
        enable_post_processing: Apply deduplication and code filtering (default: False)
        deduplicate: Remove duplicate entities within documents (default: True)
        filter_code: Filter code/technical context patterns (default: True)

    Returns:
        Dict with metrics and predictions (includes both exact and hierarchical)
    """
    # Metrics for hierarchical matching
    tp_hier, fp_hier, fn_hier = 0, 0, 0
    # Metrics for exact matching
    tp_exact, fp_exact, fn_exact = 0, 0, 0

    # Post-processing statistics
    total_removed = 0
    total_original = 0

    start_time = time.time()
    all_predictions = []

    # Use tqdm for progress bar
    iterator = tqdm(enumerate(test_data), total=len(test_data), desc="  Evaluating", disable=not verbose)

    for idx, sample in iterator:
        text = sample['text']

        # Ground truth - normalize all labels consistently
        # Handle both 'entities' and 'normalized_entities' formats
        entities_key = 'normalized_entities' if 'normalized_entities' in sample else 'entities'
        ground_truth = set()
        for ent in sample[entities_key]:
            # Handle different entity formats
            if 'entity' in ent:
                # Balanced dataset format: {'entity': '...', 'types': ['...']}
                entity_text = ent['entity'].lower().strip()
                raw_label = ent.get('types', [ent.get('type', '')])[0].lower()
            else:
                # Gold dataset format: {'text': '...', 'label': '...'}
                entity_text = ent['text'].lower().strip()
                raw_label = ent['label'].lower()

            # Normalize it
            entity_type = normalizer.normalize(raw_label, model_name)
            ground_truth.add((entity_text, entity_type))

        # Predictions - normalize consistently
        predictions = set()
        raw_predictions = []
        try:
            # Always use low threshold to capture all predictions
            # Threshold filtering happens in analysis notebook
            pred_entities = model.predict_entities(text, labels, threshold=0.1)

            # Apply post-processing if enabled
            if enable_post_processing:
                original_count = len(pred_entities)
                pred_entities = post_process_predictions(
                    pred_entities,
                    deduplicate=deduplicate,
                    filter_code=filter_code
                )
                total_original += original_count
                total_removed += (original_count - len(pred_entities))

            for pred in pred_entities:
                entity_text = pred['text'].lower().strip()
                # Normalize model's label to canonical form
                entity_type = normalizer.normalize(pred['label'].lower(), model_name)
                predictions.add((entity_text, entity_type))
                raw_predictions.append({
                    'text': pred['text'],
                    'label': pred['label'],
                    'score': pred['score'],
                    'normalized_label': entity_type
                })
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Error on sample {idx}: {e}")
            continue

        # Store sample predictions with normalized ground truth
        gt_with_normalized = []
        for ent in sample[entities_key]:
            # Handle different entity formats
            if 'entity' in ent:
                entity_text = ent['entity']
                raw_label = ent.get('types', [ent.get('type', '')])[0]
            else:
                entity_text = ent['text']
                raw_label = ent['label']
            normalized_label = normalizer.normalize(raw_label.lower(), model_name)
            gt_with_normalized.append({
                'entity': entity_text,
                'type': raw_label,  # Original label
                'normalized_type': normalized_label  # Normalized label
            })

        all_predictions.append({
            'sample_idx': idx,
            'text': text,
            'predictions': raw_predictions,
            'ground_truth': gt_with_normalized
        })

        # Metrics with hierarchical matching
        matched_preds_hier = set()
        matched_gts_hier = set()

        # Metrics with exact matching
        matched_preds_exact = set()
        matched_gts_exact = set()

        # For each prediction, check if it matches any ground truth
        for pred_text, pred_label in predictions:
            for gt_text, gt_label in ground_truth:
                if pred_text == gt_text:
                    # Hierarchical match (exact OR same family)
                    if labels_match(pred_label, gt_label):
                        matched_preds_hier.add((pred_text, pred_label))
                        matched_gts_hier.add((gt_text, gt_label))

                    # Exact match only
                    if pred_label == gt_label:
                        matched_preds_exact.add((pred_text, pred_label))
                        matched_gts_exact.add((gt_text, gt_label))
                    break

        # Update hierarchical counts
        tp_hier += len(matched_preds_hier)
        fp_hier += len(predictions) - len(matched_preds_hier)
        fn_hier += len(ground_truth) - len(matched_gts_hier)

        # Update exact counts
        tp_exact += len(matched_preds_exact)
        fp_exact += len(predictions) - len(matched_preds_exact)
        fn_exact += len(ground_truth) - len(matched_gts_exact)

    elapsed = time.time() - start_time

    # Calculate hierarchical metrics
    precision_hier = tp_hier / (tp_hier + fp_hier) if (tp_hier + fp_hier) > 0 else 0
    recall_hier = tp_hier / (tp_hier + fn_hier) if (tp_hier + fn_hier) > 0 else 0
    f1_hier = 2 * (precision_hier * recall_hier) / (precision_hier + recall_hier) if (precision_hier + recall_hier) > 0 else 0

    # Calculate exact metrics
    precision_exact = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) > 0 else 0
    recall_exact = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) > 0 else 0
    f1_exact = 2 * (precision_exact * recall_exact) / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0

    result = {
        # Hierarchical matching (default - recommended)
        'precision': precision_hier,
        'recall': recall_hier,
        'f1': f1_hier,
        'tp': tp_hier,
        'fp': fp_hier,
        'fn': fn_hier,

        # Exact matching (strict)
        'exact_precision': precision_exact,
        'exact_recall': recall_exact,
        'exact_f1': f1_exact,
        'exact_tp': tp_exact,
        'exact_fp': fp_exact,
        'exact_fn': fn_exact,

        # Improvement from hierarchical matching
        'hierarchical_improvement': f1_hier - f1_exact,

        # Other metrics
        'time': elapsed,
        'num_labels': len(labels),
        'predictions': all_predictions
    }

    # Add post-processing stats if enabled
    if enable_post_processing:
        result['post_processing'] = {
            'enabled': True,
            'deduplicate': deduplicate,
            'filter_code': filter_code,
            'total_predictions': total_original,
            'predictions_removed': total_removed,
            'predictions_kept': total_original - total_removed,
            'removal_rate': (total_removed / total_original * 100) if total_original > 0 else 0
        }
    else:
        result['post_processing'] = {'enabled': False}

    return result


def save_predictions(model_name, result, labels, threshold=0.3, output_dir='preds'):
    """Save predictions to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Handle threshold being dict or float
    threshold_info = threshold if not isinstance(threshold, dict) else "per-label"

    pred_output = {
        'model': model_name,
        'test_samples': len(result['predictions']),
        'labels_used': labels,
        'threshold': threshold_info,
        'threshold_details': threshold if isinstance(threshold, dict) else None,
        'predictions': result['predictions']
    }

    # Generate filename from model name
    filename = f"predictions_{model_name.lower().replace('-', '_')}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(pred_output, f, indent=2)

    return filepath


def save_results(results, group_name, labels, threshold=0.3, output_dir='results'):
    """Save results (metrics only) to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Remove predictions from results for cleaner output
    clean_results = {
        name: {k: v for k, v in metrics.items() if k != 'predictions'}
        for name, metrics in results.items()
    }

    # Handle threshold being dict or float
    threshold_info = threshold if not isinstance(threshold, dict) else "per-label"

    output = {
        'test_samples': len(list(results.values())[0]['predictions']) if results else 0,
        'labels_used': labels,
        'threshold': threshold_info,
        'threshold_details': threshold if isinstance(threshold, dict) else None,
        'results': clean_results
    }

    filename = f"results_{group_name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    return filepath


def display_model_results(model_name, metrics):
    """Display results for a single model"""
    print(f"\n{model_name}:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"  Time:      {metrics['time']:.1f}s")


def display_comparative_results(all_results):
    """Display comparative results table with both exact and hierarchical matching"""
    if not all_results:
        print("\n❌ No results to display")
        return

    # Sort by hierarchical F1 score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)

    # Display hierarchical results (recommended)
    print(f"\n{'=' * 100}")
    print("HIERARCHICAL MATCHING RESULTS (Recommended)")
    print('=' * 100)
    print("Uses 13 semantic families (NAME, LOCATION, MEDICAL, etc.) for matching")
    print("Example: 'first name' matches 'name', 'SSN' matches 'social security number'")
    print('-' * 100)
    print(f"{'MODEL':<25} {'LABELS':<10} {'PRECISION':<12} {'RECALL':<12} {'F1':<12} {'TIME(s)':<10}")
    print('=' * 100)

    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics['num_labels']:<10} "
              f"{metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
              f"{metrics['f1']:<12.3f} {metrics['time']:<10.1f}")

    print('=' * 100)

    # Display exact matching results (for comparison)
    print(f"\n{'=' * 100}")
    print("EXACT MATCHING RESULTS (For Reference)")
    print('=' * 100)
    print("Requires exact label match - more strict than hierarchical")
    print('-' * 100)
    print(f"{'MODEL':<25} {'LABELS':<10} {'PRECISION':<12} {'RECALL':<12} {'F1':<12} {'TIME(s)':<10}")
    print('=' * 100)

    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics['num_labels']:<10} "
              f"{metrics['exact_precision']:<12.3f} {metrics['exact_recall']:<12.3f} "
              f"{metrics['exact_f1']:<12.3f} {metrics['time']:<10.1f}")

    print('=' * 100)
