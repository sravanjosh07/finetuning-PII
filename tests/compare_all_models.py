"""
Main evaluation script - evaluates GLiNER PII models

Simple configuration: Just comment/uncomment the models you want to test below!
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import utils
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import json
import random
from gliner import GLiNER
from utils.models_config import MODELS
from utils.evaluate import (
    evaluate_model,
    save_predictions,
    save_results,
    display_comparative_results
)

# ============================================================================
# CONFIGURATION - Edit these settings
# ============================================================================

# Which models to evaluate? Comment out (#) the ones you don't want
MODELS_TO_RUN = [
    'Finetuned-PII',      # Our custom model
    'E3-JSI-Domains',     # Multi-domain PII (9 languages)
    # 'Urchade-PII',        # Urchade multilingual
    # Gretel family (large → base → small)
    # 'Gretel-Large',       # Gretel large
    'Gretel-Base',        # Gretel base
    # 'Gretel-Small',       # Gretel small
    # Knowledgator family (large → base → small)
    # 'Knowledge-Large',    # Knowledgator large
    'Knowledge-Base',     # Knowledgator base
    # 'Knowledge-Small',    # Knowledgator small
    # Other models
    # 'NVIDIA-PII',       # NVIDIA model
]

# Test data settings
# Build path relative to script location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "Data"
TEST_FILE = str(DATA_DIR / 'gold_testdataset_27labels.ndjson')  # Gold test dataset (27 labels)
SAMPLE_LIMIT = None  # Set to 0 or None to use all samples

# Model settings
THRESHOLD = 0.3  # Confidence threshold for predictions

# Post-processing settings (NEW!)
ENABLE_POST_PROCESSING = False  # Set to True to enable deduplication and code filtering
DEDUPLICATE = True              # Remove duplicate entities within documents
FILTER_CODE = True              # Filter code/technical context patterns

# Output directory for results
OUTPUT_DIR = str(SCRIPT_DIR.parent / 'results')  # Directory to save evaluation results

# ============================================================================
# Main evaluation code (you don't need to edit below this)
# ============================================================================

def load_test_data(filepath, sample_limit=None):
    """Load test data (optionally sample a subset)"""
    data = []

    # Handle both JSON and NDJSON formats
    if filepath.endswith('.ndjson'):
        # NDJSON format (newline-delimited JSON)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]  # Remove trailing comma
                if line:
                    data.append(json.loads(line))
    else:
        # Regular JSON format
        with open(filepath, 'r') as f:
            data = json.load(f)

    if sample_limit and sample_limit > 0 and sample_limit < len(data):
        random.seed(42)  # For reproducibility
        data = random.sample(data, sample_limit)
        print(f"  Randomly sampled {sample_limit} samples for quick testing")

    return data


def main():
    print("=" * 100)
    print("GLINER PII MODELS EVALUATION")
    print("=" * 100)

    # Show configuration
    print(f"\nSettings:")
    print(f"  Test file: {TEST_FILE}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Sample limit: {SAMPLE_LIMIT if SAMPLE_LIMIT else 'ALL'}")
    print(f"  Models to run: {', '.join(MODELS_TO_RUN)}")
    print(f"  Post-processing: {'ENABLED' if ENABLE_POST_PROCESSING else 'DISABLED'}")
    if ENABLE_POST_PROCESSING:
        print(f"    - Deduplication: {DEDUPLICATE}")
        print(f"    - Code filtering: {FILTER_CODE}")

    # Load test data
    print(f"\n{'=' * 100}")
    print("LOADING TEST DATA")
    print('=' * 100)
    test_data = load_test_data(TEST_FILE, sample_limit=SAMPLE_LIMIT)
    print(f"Loaded {len(test_data)} samples")

    # Run evaluation for each model
    print(f"\n{'=' * 100}")
    print("RUNNING EVALUATIONS")
    print('=' * 100)

    all_results = {}

    for model_name in MODELS_TO_RUN:
        if model_name not in MODELS:
            print(f"\n⚠️  Model '{model_name}' not found in config, skipping...")
            continue

        config = MODELS[model_name]

        print(f"\n{'=' * 80}")
        print(f"{model_name}: {config['description']}")
        print('=' * 80)

        # Load model
        print(f"  Loading model from {config['path']}...")
        try:
            model = GLiNER.from_pretrained(config['path'])
            print(f"  ✓ Loaded ({len(config['labels'])} labels)")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        # Evaluate
        print(f"  Evaluating on {len(test_data)} samples with {len(config['labels'])} labels...")
        if ENABLE_POST_PROCESSING:
            print(f"  Post-processing: Deduplication={DEDUPLICATE}, Filter Code={FILTER_CODE}")
        result = evaluate_model(
            model,
            model_name,
            config['labels'],  # Use each model's native labels
            test_data,
            threshold=THRESHOLD,
            verbose=True,
            enable_post_processing=ENABLE_POST_PROCESSING,
            deduplicate=DEDUPLICATE,
            filter_code=FILTER_CODE
        )

        all_results[model_name] = result

        # Show results - both exact and hierarchical
        print(f"\n  Results (Hierarchical Matching):")
        print(f"    F1:        {result['f1']:.3f}")
        print(f"    Precision: {result['precision']:.3f}")
        print(f"    Recall:    {result['recall']:.3f}")
        print(f"\n  Results (Exact Matching):")
        print(f"    F1:        {result['exact_f1']:.3f}")
        print(f"    Precision: {result['exact_precision']:.3f}")
        print(f"    Recall:    {result['exact_recall']:.3f}")
        print(f"\n  Improvement: +{result['hierarchical_improvement']:.3f} F1 ({result['hierarchical_improvement']*100:.1f}%)")
        print(f"  Time:        {result['time']:.1f}s")

        # Show post-processing stats if enabled
        if ENABLE_POST_PROCESSING and 'post_processing' in result:
            pp_stats = result['post_processing']
            if pp_stats['enabled']:
                print(f"\n  Post-processing:")
                print(f"    Removed:   {pp_stats['predictions_removed']} predictions ({pp_stats['removal_rate']:.1f}%)")
                print(f"    Kept:      {pp_stats['predictions_kept']} predictions")

        # Save predictions
        pred_file = save_predictions(model_name, result, config['labels'], threshold=THRESHOLD)
        print(f"  ✓ Saved predictions to {pred_file}")

    # Show comparison
    if all_results:
        display_comparative_results(all_results)

        # Save results
        print(f"\n{'=' * 100}")
        print("SAVING RESULTS")
        print('=' * 100)

        # Save by group
        groups_saved = set()
        for model_name in all_results:
            config = MODELS[model_name]
            group = config['group']

            if group not in groups_saved:
                group_results = {
                    name: result for name, result in all_results.items()
                    if MODELS[name]['group'] == group
                }
                result_file = save_results(
                    group_results,
                    group,
                    config['labels'],
                    threshold=THRESHOLD,
                    output_dir=OUTPUT_DIR
                )
                print(f"✓ Saved {group} results to {result_file}")
                groups_saved.add(group)

        # Save combined results
        combined = {
            'models_evaluated': len(all_results),
            'test_samples': len(test_data),
            'threshold': THRESHOLD,
            'normalization': '137 canonical labels + 13 hierarchical families',
            'results': {name: {k: v for k, v in metrics.items() if k != 'predictions'}
                       for name, metrics in all_results.items()}
        }

        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        combined_path = os.path.join(OUTPUT_DIR, 'results_combined.json')
        with open(combined_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"✓ Saved combined results to {combined_path}")

        print(f"\n{'=' * 100}")
        print("EVALUATION COMPLETE ✓")
        print('=' * 100)
        print(f"\nResults saved in: {OUTPUT_DIR}/")
        print(f"Predictions saved in: preds/")
        print()
    else:
        print("\n❌ No models were successfully evaluated")


if __name__ == "__main__":
    main()
