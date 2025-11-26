"""
Download Missing PII/NER Datasets

Run this script to download:
1. urchade/synthetic-pii-ner-mistral (~5k samples)
2. beki/privy (~5k samples)
3. AmitFinkman/PII_dataset (~4k samples)
4. Few-NERD (188k samples)

Usage:
    pip install datasets requests
    python download_datasets.py
"""

import os
import json
import requests
from datasets import load_dataset

# Output directory
OUTPUT_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data/additional_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Downloading PII/NER Datasets")
print("=" * 60)


# ============================================================
# 1. urchade/synthetic-pii-ner-mistral (GLiNER training data)
# ============================================================
print("\n[1/4] Downloading urchade/synthetic-pii-ner-mistral...")
try:
    dataset = load_dataset("urchade/synthetic-pii-ner-mistral-v1")

    # Convert to list and save
    data = [item for item in dataset['train']]
    output_path = os.path.join(OUTPUT_DIR, "urchade_synthetic_pii_ner_mistral.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   ✓ Saved {len(data)} samples to {output_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# 2. beki/privy (API/Protocol PII)
# ============================================================
print("\n[2/4] Downloading beki/privy...")
try:
    dataset = load_dataset("beki/privy")

    # Convert to list and save
    data = [item for item in dataset['train']]
    output_path = os.path.join(OUTPUT_DIR, "beki_privy.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   ✓ Saved {len(data)} samples to {output_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# 3. AmitFinkman/PII_dataset (GitHub - Medical, Legal, Finance)
# ============================================================
print("\n[3/4] Downloading AmitFinkman/PII_dataset from GitHub...")
try:
    # This dataset is on GitHub, need to fetch raw files
    base_url = "https://raw.githubusercontent.com/AmitFinkman/PII_dataset/main"
    files_to_download = [
        "Financial_Prompts.json",
        "Medical_Prompts.json",
        "Legal_Prompts.json",
        "Education_Prompts.json"
    ]

    all_data = []
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Add domain info
            domain = filename.replace("_Prompts.json", "").lower()
            for item in data:
                item['domain'] = domain
            all_data.extend(data)
            print(f"   ✓ Downloaded {filename}: {len(data)} samples")
        else:
            print(f"   ✗ Failed to download {filename}")

    output_path = os.path.join(OUTPUT_DIR, "amitfinkman_pii_dataset.json")
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"   ✓ Total saved: {len(all_data)} samples to {output_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# 4. Few-NERD (Large-scale fine-grained NER)
# ============================================================
print("\n[4/4] Downloading Few-NERD (this may take a while - 188k samples)...")
try:
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    # Save train split (largest)
    train_data = [item for item in dataset['train']]
    output_path = os.path.join(OUTPUT_DIR, "fewnerd_train.json")
    with open(output_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"   ✓ Saved train: {len(train_data)} samples")

    # Save test split
    test_data = [item for item in dataset['test']]
    output_path_test = os.path.join(OUTPUT_DIR, "fewnerd_test.json")
    with open(output_path_test, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"   ✓ Saved test: {len(test_data)} samples")

    print(f"   ✓ Total Few-NERD: {len(train_data) + len(test_data)} samples")
except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Download Complete!")
print("=" * 60)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nDownloaded files:")
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.json'):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
