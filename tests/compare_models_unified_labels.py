"""
Fair model comparison - All models use the same 28 updated labels from Finetuned-PII

This tests which model architecture performs best when given identical labels,
removing the bias of each model using its own native label set.

Uses UPDATED_FINETUNED_LABELS (28 labels) - cleaned up version with:
- "date" instead of "date of birth" (covers all date types)
- Removed non-PII labels (year, city, location, person, description, etc.)
- Kept "organization" (significant in test data)
"""
import os
import sys
import warnings
import json
import random
from pathlib import Path

# Add parent directory to path so we can import utils
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
warnings.filterwarnings('ignore')

# Suppress stderr during imports
import io
stderr_backup = sys.stderr
sys.stderr = io.StringIO()

from gliner import GLiNER
from utils.models_config import MODELS, UPDATED_FINETUNED_LABELS
from utils.evaluate import (
    evaluate_model,
    save_predictions,
    save_results,
    display_comparative_results
)

# Restore stderr
sys.stderr = stderr_backup

# Context manager to suppress output during model loading
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Temporarily suppress stdout and stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which models to evaluate?
MODELS_TO_RUN = [
    'Finetuned-PII',      # Our custom model
    'E3-JSI-Domains',     # Multi-domain PII (9 languages)
    'Urchade-PII',        # Urchade multilingual
    # Gretel family (large → base → small)
    'Gretel-Large',       # Gretel large
    'Gretel-Base',        # Gretel base
    'Gretel-Small',       # Gretel small
    # Knowledgator family (large → base → small)
    'Knowledge-Large',    # Knowledgator large
    'Knowledge-Base',     # Knowledgator base
    'Knowledge-Small',    # Knowledgator small
    # Other models
    'NVIDIA-PII',       # NVIDIA model
]

# Use UPDATED_FINETUNED labels for ALL models (fair comparison)
UNIFIED_LABELS = UPDATED_FINETUNED_LABELS

# Test data settings
TEST_FILE = 'balanced_test_data/datasets/balanced_test_100_per_class_27_labels_filtered_300tok.json'  # Balanced test data with 27 labels
SAMPLE_LIMIT = None  # Set to a number to test on subset, None for all

# Model settings
THRESHOLD = 0.7

# Output directories for this experiment
RESULTS_DIR = 'balanced_test_data/unified_results'
PREDS_DIR = 'balanced_test_data/unified_preds'

# ============================================================================
# MAIN CODE
# ============================================================================

def load_test_data(filepath, sample_limit=None):
    """Load test data (optionally sample a subset)"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if sample_limit and sample_limit > 0 and sample_limit < len(data):
        random.seed(42)
        data = random.sample(data, sample_limit)
        print(f"  Randomly sampled {sample_limit} samples for quick testing")

    return data


def main():
    print("=" * 100)
    print("FAIR MODEL COMPARISON - UPDATED LABELS EXPERIMENT")
    print("=" * 100)
    print(f"\nAll models will be tested using the SAME {len(UNIFIED_LABELS)} updated labels from Finetuned-PII")
    print("This removes label bias and shows which model architecture performs best.\n")

    # Show configuration
    print(f"Settings:")
    print(f"  Test file: {TEST_FILE}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Sample limit: {SAMPLE_LIMIT if SAMPLE_LIMIT else 'ALL'}")
    print(f"  Models to run: {', '.join(MODELS_TO_RUN)}")
    print(f"  Unified labels: {len(UNIFIED_LABELS)} labels (from Finetuned-PII)")
    print(f"  Results directory: {RESULTS_DIR}/")
    print(f"  Predictions directory: {PREDS_DIR}/")

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

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
            with suppress_output():
                model = GLiNER.from_pretrained(config['path'])
            print(f"  ✓ Loaded")
            print(f"  ⚠️  Using {len(UNIFIED_LABELS)} unified labels (not native {len(config['labels'])} labels)")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        # Evaluate
        print(f"  Evaluating on {len(test_data)} samples...")
        result = evaluate_model(
            model,
            model_name,
            UNIFIED_LABELS,  # Use unified labels instead of native labels
            test_data,
            threshold=THRESHOLD,
            verbose=True
        )

        all_results[model_name] = result

        # Show results
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

        # Save predictions
        pred_file = save_predictions(model_name, result, UNIFIED_LABELS, threshold=THRESHOLD, output_dir=PREDS_DIR)
        print(f"  ✓ Saved predictions to {pred_file}")

    # Show comparison
    if all_results:
        display_comparative_results(all_results)

        # Save results
        print(f"\n{'=' * 100}")
        print("SAVING RESULTS")
        print('=' * 100)

        # Save combined results
        combined = {
            'experiment': 'updated_labels',
            'description': f'All models tested with same {len(UNIFIED_LABELS)} updated labels from Finetuned-PII (cleaned)',
            'unified_labels': UNIFIED_LABELS,
            'num_labels': len(UNIFIED_LABELS),
            'models_evaluated': len(all_results),
            'test_samples': len(test_data),
            'threshold': THRESHOLD,
            'normalization': '137 canonical labels + 13 hierarchical families',
            'results': {name: {k: v for k, v in metrics.items() if k != 'predictions'}
                       for name, metrics in all_results.items()}
        }

        result_file = os.path.join(RESULTS_DIR, 'results_updated_labels.json')
        with open(result_file, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"✓ Saved results to {result_file}")

        print(f"\n{'=' * 100}")
        print("EVALUATION COMPLETE ✓")
        print('=' * 100)
        print(f"\nResults saved in: {RESULTS_DIR}/")
        print(f"Predictions saved in: {PREDS_DIR}/")
        print()
    else:
        print("\n❌ No models were successfully evaluated")


if __name__ == "__main__":
    main()
