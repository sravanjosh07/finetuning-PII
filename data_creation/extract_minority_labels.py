"""
Extract Minority Label Samples - Modular Files

Creates separate files for each underrepresented label.
Easy to combine with gold dataset or use individually.

Output format matches: balanced_test_100_per_class_27_labels_filtered_300tok.json
Each file can be used independently or combined.

Usage:
    python data_creation/extract_minority_labels.py
"""
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Minority labels to extract (labels with <100 samples in gold dataset)
MINORITY_LABELS = {
    "medical treatment": 100,      # Target: 100 samples (current: 11 in gold)
    "medication": 50,              # Target: 50 samples (current: 27 in gold)
    "amount": 100,                 # Target: 100 samples (current: 58 in gold)
    "mobile phone number": 50,     # Target: 50 samples (current: 83 in gold)
    "insurance plan number": 50,   # Target: 50 samples (current: 73 in gold)
}

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"
OUTPUT_DIR = PROJECT_DIR / "data" / "minority_labels"

# Datasets to search (medical datasets for medical labels, e3jsi for others)
MEDICAL_DATASETS = [
    (DATA_DIR / "medical_testing" / "e3jsi_english_all.json", "e3jsi-english"),
    (DATA_DIR / "medical_testing" / "e3jsi_train.json", "e3jsi-train"),
]

# Label mapping (from previous extract_medical_samples.py)
LABEL_MAPPING = {
    # Medical
    "medical treatment": "medical treatment",
    "treatment": "medical treatment",
    "medical procedure": "medical treatment",
    "procedure": "medical treatment",
    "therapy": "medical treatment",
    "surgery": "medical treatment",
    "medication": "medication",
    "drug": "medication",
    "prescription": "medication",

    # Amount
    "amount": "amount",
    "balance": "amount",
    "bedrag": "amount",
    "salary": "amount",
    "bank account balance": "amount",

    # Phone
    "mobile phone number": "mobile phone number",
    "mobile": "mobile phone number",
    "phone": "phone number",  # Different from mobile
    "phone number": "phone number",

    # Insurance
    "insurance plan number": "insurance plan number",
    "medical record number": "insurance plan number",

    # Common entities (keep for context)
    "person": "full name",
    "name": "full name",
    "date": "date",
    "organization": "organization",
    "address": "address",
    "email": "email address",
}


# ============================================================================
# LOAD AND PARSE DATASETS
# ============================================================================

def parse_e3jsi_sample(sample):
    """Parse e3jsi format sample"""
    # Skip non-English
    if sample.get('language', 'english').lower() != 'english':
        return None

    text = sample.get('text', '')
    entities = sample.get('entities', [])

    if not text or not entities:
        return None

    # Normalize entities
    normalized = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue

        entity_text = ent.get('text', ent.get('entity', ''))
        entity_type = ent.get('types', [ent.get('type', '')])
        if isinstance(entity_type, list):
            entity_type = entity_type[0] if entity_type else ''

        entity_type_lower = str(entity_type).lower().strip()

        # Map to target label
        if entity_type_lower in LABEL_MAPPING:
            mapped_label = LABEL_MAPPING[entity_type_lower]
            normalized.append({
                'entity': entity_text,
                'types': [mapped_label],
                'start': ent.get('start', 0),
                'end': ent.get('end', 0),
            })

    if not normalized:
        return None

    return {
        'text': text,
        'entities': normalized,
    }


# ============================================================================
# EXTRACT SAMPLES BY LABEL
# ============================================================================

def extract_samples_for_label(all_samples, target_label, num_samples):
    """
    Extract samples containing a specific label.

    Returns samples in balanced dataset format:
    {
        'text': '...',
        'entities': [{'entity': '...', 'types': ['...'], 'start': 0, 'end': 0}]
    }
    """
    matching = []

    for sample in all_samples:
        # Check if sample has target label
        has_target = False
        for entity in sample.get('entities', []):
            if target_label in entity.get('types', []):
                has_target = True
                break

        if has_target:
            matching.append(sample)

            if len(matching) >= num_samples:
                break

    return matching


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EXTRACT MINORITY LABEL SAMPLES - MODULAR FILES")
    print("=" * 80)
    print(f"\n📋 Target labels: {', '.join(MINORITY_LABELS.keys())}")

    # ========== Step 1: Load datasets ==========
    print(f"\n{'=' * 80}")
    print("STEP 1: LOAD DATASETS")
    print('=' * 80)

    all_samples = []

    for file_path, name in MEDICAL_DATASETS:
        if not file_path.exists():
            print(f"\n⚠️  {name}: File not found")
            continue

        print(f"\n📂 {name}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        parsed = 0
        for sample in tqdm(data, desc="   Parsing", leave=False):
            parsed_sample = parse_e3jsi_sample(sample)
            if parsed_sample:
                all_samples.append(parsed_sample)
                parsed += 1

        print(f"   ✓ Parsed {parsed} samples")

    print(f"\n✓ Total samples loaded: {len(all_samples)}")

    # ========== Step 2: Count available labels ==========
    print(f"\n{'=' * 80}")
    print("STEP 2: COUNT AVAILABLE LABELS")
    print('=' * 80)

    available_counts = defaultdict(int)
    for sample in all_samples:
        labels_in_sample = set()
        for entity in sample.get('entities', []):
            for label in entity.get('types', []):
                labels_in_sample.add(label)

        for label in labels_in_sample:
            available_counts[label] += 1

    print(f"\n{'Label':<30} {'Available':>12} {'Target':>10} {'Status'}")
    print("-" * 65)
    for label, target in MINORITY_LABELS.items():
        available = available_counts.get(label, 0)
        status = "✅" if available >= target else "⚠️"
        print(f"{label:<30} {available:>12} {target:>10} {status}")

    # ========== Step 3: Extract and save ==========
    print(f"\n{'=' * 80}")
    print("STEP 3: EXTRACT AND SAVE")
    print('=' * 80)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    extracted_stats = {}

    for label, target_count in MINORITY_LABELS.items():
        print(f"\n📦 Extracting: {label}")

        # Extract samples
        samples = extract_samples_for_label(all_samples, label, target_count)

        if not samples:
            print(f"   ⚠️  No samples found")
            continue

        # Save to file
        safe_filename = label.replace(' ', '_')
        output_file = OUTPUT_DIR / f"{safe_filename}_samples.json"

        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)

        # Count entities
        entity_count = sum(
            1 for s in samples
            for e in s.get('entities', [])
            if label in e.get('types', [])
        )

        extracted_stats[label] = {
            'samples': len(samples),
            'entities': entity_count,
            'file': str(output_file.name)
        }

        print(f"   ✓ {len(samples)} samples, {entity_count} entities")
        print(f"   ✓ Saved to: {output_file.name}")

    # ========== Step 4: Create index ==========
    print(f"\n{'=' * 80}")
    print("STEP 4: CREATE INDEX FILE")
    print('=' * 80)

    index = {
        'description': 'Minority label samples - modular files',
        'usage': 'Load individual files or combine as needed',
        'format': 'Same as balanced_test_100_per_class_27_labels_filtered_300tok.json',
        'files': extracted_stats
    }

    index_file = OUTPUT_DIR / 'README.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n✓ Created index: {index_file.name}")

    # ========== Summary ==========
    print(f"\n{'=' * 80}")
    print("✅ COMPLETE!")
    print('=' * 80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\n📋 Files created:")
    for label, stats in extracted_stats.items():
        print(f"   • {stats['file']:<40} {stats['samples']:>4} samples, {stats['entities']:>4} entities")

    print(f"\n💡 Usage:")
    print(f"   # Load individual file")
    print(f"   with open('data/minority_labels/medication_samples.json') as f:")
    print(f"       medication_data = json.load(f)")
    print(f"   ")
    print(f"   # Combine multiple files")
    print(f"   combined = medication_data + medical_treatment_data + amount_data")
    print(f"   ")
    print(f"   # Add to gold dataset")
    print(f"   augmented = gold_dataset + minority_samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
