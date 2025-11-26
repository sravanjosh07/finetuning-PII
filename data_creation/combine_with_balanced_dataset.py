"""
Combine with Balanced Dataset

Combines balanced_test_100_per_class_27_labels_filtered_300tok.json with minority label samples.
Outputs in the same JSON format with matching keys.

Usage:
    python data_creation/combine_with_balanced_dataset.py
"""
import json
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"
MINORITY_DIR = PROJECT_DIR / "data" / "minority_labels"

# Input files
BALANCED_FILE = DATA_DIR / "combined_testdata" / "balanced_test_100_per_class_27_labels_filtered_300tok.json"

# Minority labels to add
MINORITY_LABELS_TO_ADD = {
    "medical_treatment_samples.json": True,
    "medication_samples.json": True,
    "amount_samples.json": True,
}

# Output
OUTPUT_FILE = PROJECT_DIR / "data" / "balanced_augmented_test_dataset.json"


# Label to canonical type mapping (for consistency)
LABEL_TO_CANONICAL = {
    "date": "date",
    "full name": "name",
    "social security number": "ssn",
    "tax identification number": "taxnum",
    "drivers license number": "driver_license",
    "identity card number": "national_id",
    "passport number": "passport",
    "birth certificate number": "birth_certificate",
    "student id number": "student_id",
    "phone number": "phone",
    "fax number": "fax_number",
    "email address": "email",
    "ip address": "ip_address",
    "address": "address",
    "credit card number": "credit_card",
    "credit score": "credit_score",
    "bank account number": "bank_account",
    "amount": "amount",
    "iban": "iban",
    "health insurance id number": "health_insurance",
    "insurance plan number": "insurance_plan",
    "national health insurance number": "national_health_insurance",
    "medical condition": "medical_condition",
    "medication": "medication",
    "medical treatment": "procedure",
    "username": "username",
    "organization": "organization",
}


def load_balanced_dataset():
    """Load balanced test dataset"""
    with open(BALANCED_FILE, 'r') as f:
        data = json.load(f)
    return data


def load_minority_samples(filename):
    """Load minority label samples"""
    filepath = MINORITY_DIR / filename
    if not filepath.exists():
        print(f"   ⚠️  {filename} not found")
        return []

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def convert_to_balanced_format(sample):
    """
    Convert minority label format to balanced dataset format.

    Input format:
        {'text': '...', 'entities': [{'entity': '...', 'types': [...], 'start': 0, 'end': 0}]}

    Output format:
        {'text': '...', 'entities': [{'entity': '...', 'types': [...], 'start': 0, 'end': 0,
                                       'original_type': '...', 'canonical_type': '...'}],
         'source_dataset': 'minority_labels'}
    """
    converted = {
        'text': sample['text'],
        'entities': [],
        'source_dataset': 'minority_labels'
    }

    for entity in sample.get('entities', []):
        # Get label
        entity_types = entity.get('types', [])
        if isinstance(entity_types, list) and entity_types:
            label = entity_types[0]
        else:
            label = str(entity_types)

        # Merge mobile phone → phone number
        if label == 'mobile phone number':
            label = 'phone number'
        elif label == 'bank account balance':
            label = 'amount'

        # Get canonical type
        canonical_type = LABEL_TO_CANONICAL.get(label, label.replace(' ', '_'))

        converted['entities'].append({
            'entity': entity.get('entity', ''),
            'types': [label],
            'start': entity.get('start', 0),
            'end': entity.get('end', 0),
            'original_type': canonical_type,
            'canonical_type': canonical_type
        })

    return converted


def deduplicate_samples(samples):
    """Remove duplicates by text"""
    seen_texts = set()
    unique_samples = []

    for sample in samples:
        text = sample['text']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_samples.append(sample)

    return unique_samples


def main():
    print("=" * 80)
    print("COMBINE WITH BALANCED DATASET")
    print("=" * 80)

    # Load balanced dataset
    print(f"\n📂 Loading balanced dataset: {BALANCED_FILE.name}")
    balanced_samples = load_balanced_dataset()
    print(f"   ✓ {len(balanced_samples)} samples")

    # Start with balanced samples
    all_samples = balanced_samples.copy()

    # Add minority labels
    print(f"\n📦 Adding minority label samples:")
    print("-" * 80)

    for filename, should_add in MINORITY_LABELS_TO_ADD.items():
        if not should_add:
            print(f"   ⏭  {filename:<45} (skipped)")
            continue

        minority_samples = load_minority_samples(filename)
        if not minority_samples:
            continue

        # Convert to balanced format
        converted = [convert_to_balanced_format(s) for s in minority_samples]

        # Count entities by label
        label_counts = Counter()
        for sample in converted:
            for entity in sample['entities']:
                label = entity['types'][0] if entity['types'] else 'unknown'
                label_counts[label] += 1

        all_samples.extend(converted)

        label_summary = ', '.join([f"{label}: {count}" for label, count in label_counts.most_common(3)])
        print(f"   ✓ {filename:<45} {len(converted):>4} samples ({label_summary})")

    # Deduplicate
    print(f"\n🔍 Removing duplicates...")
    unique_samples = deduplicate_samples(all_samples)
    duplicates_removed = len(all_samples) - len(unique_samples)
    print(f"   ✓ Removed {duplicates_removed} duplicates")

    # Save
    print(f"\n💾 Saving augmented dataset...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(unique_samples, f, indent=2)

    print(f"   ✓ Saved to: {OUTPUT_FILE.name}")

    # Summary
    print(f"\n{'=' * 80}")
    print("✅ COMPLETE!")
    print('=' * 80)
    print(f"\nDataset statistics:")
    print(f"  Balanced dataset:    {len(balanced_samples):>6} samples")
    print(f"  Minority labels:     {len(all_samples) - len(balanced_samples):>6} samples added")
    print(f"  Duplicates removed:  {duplicates_removed:>6} samples")
    print(f"  Final dataset:       {len(unique_samples):>6} samples")

    # Count entities by label
    label_counts = Counter()
    for sample in unique_samples:
        for entity in sample.get('entities', []):
            label = entity.get('types', ['unknown'])[0]
            # Merge mobile phone → phone number for counting
            if label == 'mobile phone number':
                label = 'phone number'
            elif label == 'bank account balance':
                label = 'amount'
            label_counts[label] += 1

    print(f"\nTop 15 labels in final dataset:")
    for label, count in label_counts.most_common(15):
        print(f"  {label:<40} {count:>6}")

    print(f"\n📁 Output: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
