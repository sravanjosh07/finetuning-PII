"""
Convert all PII datasets to unified format:
{
    "source_text": "...",
    "language": "en",
    "source": "dataset_name",
    "privacy_mask": [
        {"label": "LABEL", "start": 0, "end": 10, "value": "..."}
    ]
}
"""

import json
import os
import ast

# ============================================================
# Paths
# ============================================================
DATA_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data"
OUTPUT_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data/additional_datasets/unified"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper function to save and show info
# ============================================================
def save_dataset(data, name):
    output_path = os.path.join(OUTPUT_DIR, f"{name}_unified.json")
    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Saved {len(data):,} samples ({size_mb:.2f} MB)")

    # Show sample
    for sample in data[:50]:
        if sample.get("privacy_mask"):
            print(f"   Sample text: {sample['source_text'][:80]}...")
            print(f"   Sample entities: {sample['privacy_mask'][:2]}")
            break


# ============================================================
# 1. ai4privacy-pii-masking-200k (already unified format)
# ============================================================
print("=" * 60)
print("1. ai4privacy-pii-masking-200k")
print("=" * 60)

with open(f"{DATA_DIR}/ai4privacy-pii-masking-200k/train.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    converted.append({
        "source_text": item.get("source_text", ""),
        "language": item.get("language", "en"),
        "source": "ai4privacy_200k",
        "privacy_mask": item.get("privacy_mask", [])
    })

save_dataset(converted, "ai4privacy_200k")


# ============================================================
# 2. ai4privacy_pii_masking_400k (already unified format)
# ============================================================
print("\n" + "=" * 60)
print("2. ai4privacy_pii_masking_400k")
print("=" * 60)

with open(f"{DATA_DIR}/additional_datasets/ai4privacy_pii_masking_400k.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    converted.append({
        "source_text": item.get("source_text", ""),
        "language": item.get("language", "en"),
        "source": "ai4privacy_400k",
        "privacy_mask": item.get("privacy_mask", [])
    })

save_dataset(converted, "ai4privacy_400k")


# ============================================================
# 3. beki_privy
#    Format: full_text, spans: [{entity_type, entity_value, start_position, end_position}]
# ============================================================
print("\n" + "=" * 60)
print("3. beki_privy")
print("=" * 60)

with open(f"{DATA_DIR}/additional_datasets/beki_privy.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    privacy_mask = []
    for span in item.get("spans", []):
        if span.get("entity_type") != "O":
            privacy_mask.append({
                "label": span.get("entity_type", ""),
                "start": span.get("start_position", 0),
                "end": span.get("end_position", 0),
                "value": span.get("entity_value", "")
            })

    converted.append({
        "source_text": item.get("full_text", ""),
        "language": "en",
        "source": "beki_privy",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "beki_privy")


# ============================================================
# 4. urchade_synthetic_pii_ner_mistral
#    Format: tokenized_text, ner: [[start_idx, end_idx, label], ...]
# ============================================================
print("\n" + "=" * 60)
print("4. urchade_synthetic_pii_ner_mistral")
print("=" * 60)

with open(f"{DATA_DIR}/additional_datasets/urchade_synthetic_pii_ner_mistral.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    tokens = item.get("tokenized_text", [])
    ner = item.get("ner", [])

    # Reconstruct text and track positions
    text_parts = []
    token_char_starts = []
    current_pos = 0

    for i, token in enumerate(tokens):
        token_char_starts.append(current_pos)
        text_parts.append(token)
        current_pos += len(token)

        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
            if next_token not in [',', '.', '!', '?', ':', ';', "'", '"', ')', ']', '}']:
                text_parts.append(" ")
                current_pos += 1

    text = "".join(text_parts)

    privacy_mask = []
    for span in ner:
        if len(span) >= 3:
            start_idx, end_idx, label = span[0], span[1], span[2]
            if start_idx < len(tokens) and end_idx < len(tokens):
                entity_tokens = tokens[start_idx:end_idx + 1]
                entity_text = " ".join(entity_tokens)
                char_start = token_char_starts[start_idx]

                privacy_mask.append({
                    "label": label,
                    "start": char_start,
                    "end": char_start + len(entity_text),
                    "value": entity_text
                })

    converted.append({
        "source_text": text,
        "language": "en",
        "source": "urchade_synthetic",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "urchade")


# ============================================================
# 5. e3jsi_synthetic_multi_pii
#    Format: text, entities: [{entity, types}], language
# ============================================================
print("\n" + "=" * 60)
print("5. e3jsi_synthetic_multi_pii")
print("=" * 60)

with open(f"{DATA_DIR}/additional_datasets/e3jsi_synthetic_multi_pii.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    text = item.get("text", "")
    entities = item.get("entities", [])
    language = item.get("language", "en")

    privacy_mask = []
    for ent in entities:
        entity_text = ent.get("entity", "")
        entity_types = ent.get("types", [])

        start = text.find(entity_text)
        if start != -1:
            label = entity_types[0] if entity_types else "UNKNOWN"
            privacy_mask.append({
                "label": label,
                "start": start,
                "end": start + len(entity_text),
                "value": entity_text
            })

    converted.append({
        "source_text": text,
        "language": language,
        "source": "e3jsi_synthetic",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "e3jsi")


# ============================================================
# 6. gretel-pii-masking-en-v1
#    Format: text, entities (string): [{entity, types}] - need to find positions
# ============================================================
print("\n" + "=" * 60)
print("6. gretel-pii-masking-en-v1")
print("=" * 60)

with open(f"{DATA_DIR}/gretel-pii-masking-en-v1/test.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    text = item.get("text", "")
    entities_str = item.get("entities", "[]")

    # Parse string to list of dicts
    try:
        entities = ast.literal_eval(entities_str) if isinstance(entities_str, str) else entities_str
    except:
        entities = []

    privacy_mask = []
    for ent in entities:
        entity_text = ent.get("entity", "")
        entity_types = ent.get("types", [])

        # Find position in text
        start = text.find(entity_text)
        if start != -1:
            label = entity_types[0] if entity_types else "UNKNOWN"
            privacy_mask.append({
                "label": label,
                "start": start,
                "end": start + len(entity_text),
                "value": entity_text
            })

    converted.append({
        "source_text": text,
        "language": "en",
        "source": "gretel_pii_en",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "gretel_pii_en")


# ============================================================
# 7. gretel-finance-multilingual
#    Format: generated_text, pii_spans (JSON string): [{label, start, end}]
# ============================================================
print("\n" + "=" * 60)
print("7. gretel-finance-multilingual")
print("=" * 60)

with open(f"{DATA_DIR}/gretel-finance-multilingual/test.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    text = item.get("generated_text", "")
    pii_spans_str = item.get("pii_spans", "[]")
    language = item.get("language", "en")

    # Parse JSON string
    try:
        pii_spans = json.loads(pii_spans_str) if isinstance(pii_spans_str, str) else pii_spans_str
    except:
        pii_spans = []

    privacy_mask = []
    if isinstance(pii_spans, list):
        for span in pii_spans:
            if isinstance(span, dict):
                start = span.get("start", 0)
                end = span.get("end", 0)
                # Extract value from text using positions
                value = text[start:end] if start < len(text) and end <= len(text) else ""
                privacy_mask.append({
                    "label": span.get("label", ""),
                    "start": start,
                    "end": end,
                    "value": value
                })

    converted.append({
        "source_text": text,
        "language": language,
        "source": "gretel_finance",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "gretel_finance")


# ============================================================
# 8. nvidia-nemotron-pii
#    Format: text, spans (Python dict string): [{text, label, start, end}]
# ============================================================
print("\n" + "=" * 60)
print("8. nvidia-nemotron-pii")
print("=" * 60)

with open(f"{DATA_DIR}/nvidia-nemotron-pii/test.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    text = item.get("text", "")
    spans_str = item.get("spans", "[]")

    # Parse Python dict string
    try:
        spans = ast.literal_eval(spans_str) if isinstance(spans_str, str) else spans_str
    except:
        spans = []

    privacy_mask = []
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict):
                privacy_mask.append({
                    "label": span.get("label", ""),
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                    "value": span.get("text", "")  # nvidia uses 'text' not 'value'
                })

    converted.append({
        "source_text": text,
        "language": "en",
        "source": "nvidia_nemotron",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "nvidia_nemotron")


# ============================================================
# 9. gliner_pii_train_split
#    Format: tokenized_text, ner (same as urchade)
# ============================================================
print("\n" + "=" * 60)
print("9. gliner_pii_train_split")
print("=" * 60)

with open(f"{DATA_DIR}/gliner_pii_train_split.json") as f:
    data = json.load(f)
print(f"   Loaded {len(data):,} samples")

converted = []
for item in data:
    tokens = item.get("tokenized_text", [])
    ner = item.get("ner", [])

    # Reconstruct text
    text_parts = []
    token_char_starts = []
    current_pos = 0

    for i, token in enumerate(tokens):
        token_char_starts.append(current_pos)
        text_parts.append(token)
        current_pos += len(token)

        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
            if next_token not in [',', '.', '!', '?', ':', ';', "'", '"', ')', ']', '}']:
                text_parts.append(" ")
                current_pos += 1

    text = "".join(text_parts)

    privacy_mask = []
    for span in ner:
        if len(span) >= 3:
            start_idx, end_idx, label = span[0], span[1], span[2]
            if start_idx < len(tokens) and end_idx < len(tokens):
                entity_tokens = tokens[start_idx:end_idx + 1]
                entity_text = " ".join(entity_tokens)
                char_start = token_char_starts[start_idx]

                privacy_mask.append({
                    "label": label,
                    "start": char_start,
                    "end": char_start + len(entity_text),
                    "value": entity_text
                })

    converted.append({
        "source_text": text,
        "language": "en",
        "source": "gliner_pii",
        "privacy_mask": privacy_mask
    })

save_dataset(converted, "gliner_pii")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("CONVERSION COMPLETE!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}\n")

total_samples = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('_unified.json'):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        with open(path) as file:
            data = json.load(file)
        total_samples += len(data)
        print(f"  {f:35s} {len(data):>10,} samples  ({size_mb:6.1f} MB)")

print(f"\n  {'TOTAL':35s} {total_samples:>10,} samples")
