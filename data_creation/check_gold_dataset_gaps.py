"""
Check Gold Dataset Gaps - Simple Analysis

Shows which labels are underrepresented in the gold test dataset
so we can decide what to augment.

Usage:
    python data_creation/check_gold_dataset_gaps.py
"""
import json
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"
GOLD_FILE = DATA_DIR / "gold_testdataset_27labels.ndjson"

# Our 26 labels (merged mobile phone number + phone number)
OUR_26_LABELS = [
    "date", "full name",
    "social security number", "tax identification number",
    "drivers license number", "identity card number", "passport number",
    "birth certificate number", "student id number",
    "phone number", "fax number",
    "email address", "ip address", "address",
    "credit card number", "credit score", "bank account number",
    "amount",
    "iban",
    "health insurance id number", "insurance plan number",
    "national health insurance number",
    "medical condition", "medication", "medical treatment",
    "username", "organization"
]


def load_gold_dataset():
    """Load gold test dataset"""
    samples = []

    with open(GOLD_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            if line:
                samples.append(json.loads(line))

    return samples


def main():
    print("=" * 80)
    print("GOLD DATASET GAP ANALYSIS")
    print("=" * 80)

    # Load dataset
    print(f"\n📂 Loading: {GOLD_FILE}")
    samples = load_gold_dataset()
    print(f"✓ Loaded {len(samples)} samples")

    # Count labels (raw labels from gold dataset, with mobile phone merging)
    label_counts = Counter()

    for sample in samples:
        entities = sample.get('normalized_entities', [])
        for entity in entities:
            label = entity.get('label', '').lower()
            # Merge mobile phone number → phone number
            if label == 'mobile phone number':
                label = 'phone number'
            # Map bank account balance → amount
            elif label == 'bank account balance':
                label = 'amount'
            label_counts[label] += 1

    # Analyze gaps
    print(f"\n{'=' * 80}")
    print("LABEL DISTRIBUTION")
    print('=' * 80)
    print(f"{'Label':<45} {'Count':>8} {'Status'}")
    print("-" * 80)

    well_covered = []    # >= 100
    okay = []            # 50-99
    low = []             # 20-49
    very_low = []        # 10-19
    critical = []        # < 10

    for label in OUR_26_LABELS:
        count = label_counts.get(label, 0)

        if count >= 100:
            status = "✅ Well covered"
            well_covered.append((label, count))
        elif count >= 50:
            status = "✓  Okay"
            okay.append((label, count))
        elif count >= 20:
            status = "⚠️  Low"
            low.append((label, count))
        elif count >= 10:
            status = "⚠️  Very low"
            very_low.append((label, count))
        else:
            status = "❌ Critical"
            critical.append((label, count))

        print(f"{label:<45} {count:>8,} {status}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    print(f"✅ Well covered (≥100):  {len(well_covered)} labels")
    print(f"✓  Okay (50-99):         {len(okay)} labels")
    print(f"⚠️  Low (20-49):          {len(low)} labels")
    print(f"⚠️  Very low (10-19):     {len(very_low)} labels")
    print(f"❌ Critical (<10):       {len(critical)} labels")

    # Recommendations
    print(f"\n{'=' * 80}")
    print("AUGMENTATION RECOMMENDATIONS")
    print('=' * 80)

    print(f"\n🎯 Priority 1 - Critical gaps (< 10 samples):")
    if critical:
        for label, count in sorted(critical, key=lambda x: x[1]):
            print(f"   • {label:<40} ({count} samples)")
    else:
        print("   None!")

    print(f"\n🎯 Priority 2 - Very low (10-19 samples):")
    if very_low:
        for label, count in sorted(very_low, key=lambda x: x[1]):
            print(f"   • {label:<40} ({count} samples)")
    else:
        print("   None!")

    print(f"\n💡 Consider augmenting (20-49 samples):")
    if low:
        for label, count in sorted(low, key=lambda x: x[1]):
            print(f"   • {label:<40} ({count} samples)")
    else:
        print("   None!")

    print(f"\n{'=' * 80}")
    print("NOTES")
    print('=' * 80)
    print("• Medical treatment: Now augmented! (was 0, now have 206 entities)")
    print("• Some labels rare by nature (birth certificate, student ID)")
    print("• Focus on labels that:")
    print("  - Are common in real-world PII")
    print("  - Available in our datasets")
    print("  - Worth the effort")
    print("=" * 80)


if __name__ == "__main__":
    main()
