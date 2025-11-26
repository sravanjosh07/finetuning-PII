"""
Evaluate Models with LLM-Relabeled Dataset

Uses the relabeled balanced_augmented_relabeled.ndjson as ground truth.
Tests all models and shows updated scores.

Usage:
    python tests/eval_with_relabeled.py
"""
import json
import sys
from pathlib import Path

# Add project to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.evaluate import evaluate_model
from gliner import GLiNER

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which models to test (set to False to skip)
TEST_MODELS = {
    "gliner-multitask-large-v0.5": True,  # HuggingFace model
    "finetuned (ours)": False,  # Set path below if you have a finetuned model
}

# Model paths
FINETUNED_MODEL = None  # Set to your finetuned model path if available
HF_MODEL = "urchade/gliner_multitask-large-v0.5"

# Input data
INPUT_FILE = PROJECT_DIR / "data" / "balanced_augmented_relabeled.ndjson"

# Evaluation settings
THRESHOLD = 0.3  # Detection threshold

# Our 26 labels (with phone merged)
LABELS = [
    "date", "full name",
    "social security number", "tax identification number",
    "drivers license number", "identity card number", "passport number",
    "birth certificate number", "student id number",
    "phone number", "fax number",  # mobile merged into phone
    "email address", "ip address", "address",
    "credit card number", "credit score", "bank account number",
    "amount",
    "iban",
    "health insurance id number", "insurance plan number",
    "national health insurance number",
    "medical condition", "medication", "medical treatment",
    "username", "organization"
]


def load_relabeled_data():
    """
    Load relabeled NDJSON and convert to evaluation format.

    Relabeled format:
        {
            "text": "...",
            "entities": [...],  # Relabeled by LLM (use as ground truth)
            "original_entities": [...],
            "sample_idx": 0
        }

    Evaluation format:
        {
            "text": "...",
            "entities": [
                {"entity": "...", "label": "...", "start": 0, "end": 0}
            ]
        }
    """
    data = []

    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())

                # Convert LLM entities format to evaluation format
                converted_entities = []
                for ent in sample.get("entities", []):
                    # Merge mobile phone → phone number
                    label = ent.get("label", "")
                    if label == "mobile phone number":
                        label = "phone number"

                    converted_entities.append({
                        "entity": ent.get("text", ""),
                        "label": label,
                        "start": ent.get("start", 0),
                        "end": ent.get("end", 0)
                    })

                data.append({
                    "text": sample["text"],
                    "entities": converted_entities
                })

    return data


def main():
    print("=" * 80)
    print("MODEL EVALUATION - LLM RELABELED DATASET")
    print("=" * 80)

    # Load relabeled data
    print(f"\n📂 Loading relabeled dataset: {INPUT_FILE.name}")
    test_data = load_relabeled_data()
    print(f"   ✓ Loaded {len(test_data)} samples")

    # Count entities by label
    from collections import Counter
    label_counts = Counter()
    for sample in test_data:
        for entity in sample["entities"]:
            label_counts[entity["label"]] += 1

    print(f"\n📊 Ground truth distribution (top 10):")
    for label, count in label_counts.most_common(10):
        print(f"   {label:<40} {count:>6}")

    # Test models
    print(f"\n{'=' * 80}")
    print("TESTING MODELS")
    print('=' * 80)

    results = []

    # Test HuggingFace model
    if TEST_MODELS.get("gliner-multitask-large-v0.5"):
        print(f"\n[1/1] Testing: gliner-multitask-large-v0.5")
        print(f"   Loading from HuggingFace...")

        try:
            model = GLiNER.from_pretrained(HF_MODEL)
            result = evaluate_model(
                model=model,
                model_name="gliner-multitask-large-v0.5",
                labels=LABELS,
                test_data=test_data,
                threshold=THRESHOLD,
                verbose=True
            )
            results.append(result)
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Test finetuned model (if path is set)
    if TEST_MODELS.get("finetuned (ours)") and FINETUNED_MODEL:
        print(f"\n[2/2] Testing: finetuned (ours)")
        print(f"   Loading from: {FINETUNED_MODEL}")

        try:
            # Load local model (saved with model.save_pretrained())
            model = GLiNER.from_pretrained(FINETUNED_MODEL, local_files_only=True)
            result = evaluate_model(
                model=model,
                model_name="finetuned (ours)",
                labels=LABELS,
                test_data=test_data,
                threshold=THRESHOLD,
                verbose=True
            )
            results.append(result)
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Summary
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print('=' * 80)

    if results:
        print(f"\n{'Model':<35} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['model']:<35} {r['precision']:>10.1%} {r['recall']:>10.1%} {r['f1']:>10.1%}")
    else:
        print("No models tested!")

    print(f"\n{'=' * 80}")
    print("Dataset Info:")
    print(f"  Test samples:    {len(test_data)}")
    print(f"  Total entities:  {sum(label_counts.values())}")
    print(f"  Unique labels:   {len(label_counts)}")
    print(f"  Ground truth:    LLM relabeled (merged names, normalized labels)")
    print(f"  Threshold:       {THRESHOLD}")
    print("=" * 80)


if __name__ == "__main__":
    main()
