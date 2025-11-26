"""
Download Missing PII/NER Datasets (FIXED VERSION)

Some datasets need special handling:
- urchade: download raw data.json from repo
- beki/privy: requires special loading (trust_remote_code)
- AmitFinkman: download from GitHub raw files
- Few-NERD: standard HuggingFace load

Usage:
    pip install datasets requests huggingface_hub
    python download_datasets_fixed.py
"""

import os
import json
import requests

# Output directory
OUTPUT_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data/additional_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Downloading PII/NER Datasets (Fixed Version)")
print("=" * 60)


# ============================================================
# 1. urchade/synthetic-pii-ner-mistral-v1
#    (Download raw data.json directly from HF repo)
# ============================================================
print("\n[1/4] Downloading urchade/synthetic-pii-ner-mistral-v1...")
try:
    # Direct download from HuggingFace repo (allow redirects)
    url = "https://huggingface.co/datasets/urchade/synthetic-pii-ner-mistral-v1/resolve/main/data.json"
    response = requests.get(url, allow_redirects=True, timeout=60)

    if response.status_code == 200:
        data = response.json()
        output_path = os.path.join(OUTPUT_DIR, "urchade_synthetic_pii_ner_mistral.json")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   ✓ Saved {len(data)} samples to {output_path}")
    else:
        print(f"   ✗ Failed to download. Status: {response.status_code}")
        print(f"   Try manual download from: {url}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   Try manual: https://huggingface.co/datasets/urchade/synthetic-pii-ner-mistral-v1/tree/main")


# ============================================================
# 2. beki/privy (requires trust_remote_code=True)
# ============================================================
print("\n[2/4] Downloading beki/privy...")
try:
    from datasets import load_dataset

    # This dataset requires trust_remote_code
    dataset = load_dataset("beki/privy", trust_remote_code=True)

    # Get all available splits
    all_data = []
    for split_name in dataset.keys():
        split_data = [item for item in dataset[split_name]]
        all_data.extend(split_data)
        print(f"   ✓ Loaded split '{split_name}': {len(split_data)} samples")

    output_path = os.path.join(OUTPUT_DIR, "beki_privy.json")
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"   ✓ Total saved: {len(all_data)} samples to {output_path}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    print("   Try manual download from: https://huggingface.co/datasets/beki/privy")


# ============================================================
# 3. AmitFinkman/PII_dataset (GitHub)
#    Files are in prompts_to_llm/ folder
# ============================================================
print("\n[3/4] Downloading AmitFinkman/PII_dataset from GitHub...")
try:
    base_url = "https://raw.githubusercontent.com/AmitFinkman/PII_dataset/main/prompts_to_llm"
    files_to_download = [
        ("employer_prompts_finance.json", "finance"),
        ("employer_prompts_medical.json", "medical"),
        ("employer_prompts_legal.json", "legal"),
        ("employer_prompts_education.json", "education")
    ]

    all_data = []
    for filename, domain in files_to_download:
        url = f"{base_url}/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Add domain info to each item
            for item in data:
                item['domain'] = domain
            all_data.extend(data)
            print(f"   ✓ Downloaded {filename}: {len(data)} samples")
        else:
            print(f"   ✗ Failed to download {filename} (status: {response.status_code})")

    if all_data:
        output_path = os.path.join(OUTPUT_DIR, "amitfinkman_pii_dataset.json")
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"   ✓ Total saved: {len(all_data)} samples to {output_path}")
    else:
        print("   ✗ No data downloaded")

except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# 4. Few-NERD (standard HuggingFace)
# ============================================================
print("\n[4/4] Downloading Few-NERD (this may take a while - 188k samples)...")
try:
    from datasets import load_dataset

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    # Save train split
    train_data = [dict(item) for item in dataset['train']]
    output_path = os.path.join(OUTPUT_DIR, "fewnerd_train.json")
    with open(output_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"   ✓ Saved train: {len(train_data)} samples")

    # Save test split
    test_data = [dict(item) for item in dataset['test']]
    output_path_test = os.path.join(OUTPUT_DIR, "fewnerd_test.json")
    with open(output_path_test, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"   ✓ Saved test: {len(test_data)} samples")

    # Save dev split if exists
    if 'validation' in dataset:
        dev_data = [dict(item) for item in dataset['validation']]
        output_path_dev = os.path.join(OUTPUT_DIR, "fewnerd_dev.json")
        with open(output_path_dev, 'w') as f:
            json.dump(dev_data, f, indent=2)
        print(f"   ✓ Saved validation: {len(dev_data)} samples")

    print(f"   ✓ Total Few-NERD downloaded successfully")

except Exception as e:
    print(f"   ✗ Error: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Download Complete!")
print("=" * 60)
print(f"\nFiles in: {OUTPUT_DIR}")
print("\nAll JSON files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.json'):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)

        # Count samples
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                count = len(data) if isinstance(data, list) else "N/A"
        except:
            count = "?"

        print(f"  {f:45s} {size_mb:8.2f} MB  ({count} samples)")
