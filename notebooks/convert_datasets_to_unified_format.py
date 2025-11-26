"""
Convert all datasets to unified ai4privacy format:
{
    "source_text": "...",
    "language": "en",
    "privacy_mask": [
        {"label": "LABEL", "start": 0, "end": 10, "value": "..."}
    ]
}

Datasets to convert:
1. beki_privy.json - has spans with positions
2. urchade_synthetic_pii_ner_mistral.json - has entities without positions
3. fewnerd_train.json - has tokens + numeric tags (general NER, not PII)
"""

import json
import os
import re

# ============================================================
# Paths
# ============================================================
INPUT_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data/additional_datasets"
OUTPUT_DIR = "/Users/sravan/Documents/Experiments/fintuning_PII/Data/additional_datasets/unified"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Convert beki_privy.json
#    Format: full_text, spans: [{entity_type, entity_value, start_position, end_position}]
# ============================================================
print("=" * 60)
print("1. Converting beki_privy.json...")
print("=" * 60)

with open(os.path.join(INPUT_DIR, "beki_privy.json")) as f:
    beki_data = json.load(f)

print(f"   Loaded {len(beki_data)} samples")

beki_converted = []
for item in beki_data:
    # Filter out "O" (non-entity) spans
    privacy_mask = []
    for span in item.get("spans", []):
        if span.get("entity_type") != "O":
            privacy_mask.append({
                "label": span.get("entity_type", ""),
                "start": span.get("start_position", 0),
                "end": span.get("end_position", 0),
                "value": span.get("entity_value", "")
            })

    converted = {
        "source_text": item.get("full_text", ""),
        "language": "en",
        "source": "beki_privy",
        "privacy_mask": privacy_mask
    }
    beki_converted.append(converted)

# Save
output_path = os.path.join(OUTPUT_DIR, "beki_privy_unified.json")
with open(output_path, "w") as f:
    json.dump(beki_converted, f, indent=2)
print(f"   Saved {len(beki_converted)} samples to {output_path}")

# Show sample
print("   Sample:")
print(f"   Text: {beki_converted[0]['source_text'][:100]}...")
print(f"   Entities: {beki_converted[0]['privacy_mask'][:3]}")


# ============================================================
# 2. Convert urchade_synthetic_pii_ner_mistral.json
#    Format: text, entities: [{entity, types}] - need to find positions
# ============================================================
print("\n" + "=" * 60)
print("2. Converting urchade_synthetic_pii_ner_mistral.json...")
print("=" * 60)

with open(os.path.join(INPUT_DIR, "urchade_synthetic_pii_ner_mistral.json")) as f:
    urchade_data = json.load(f)

print(f"   Loaded {len(urchade_data)} samples")
print(f"   Sample keys: {list(urchade_data[0].keys())}")

urchade_converted = []
skipped = 0

for item in urchade_data:
    text = item.get("text", "")
    entities = item.get("entities", [])

    privacy_mask = []
    for ent in entities:
        entity_text = ent.get("entity", "")
        entity_types = ent.get("types", [])

        # Find position in text
        start = text.find(entity_text)
        if start != -1:
            end = start + len(entity_text)
            # Use first type as label
            label = entity_types[0] if entity_types else "UNKNOWN"
            privacy_mask.append({
                "label": label,
                "start": start,
                "end": end,
                "value": entity_text
            })

    converted = {
        "source_text": text,
        "language": "en",
        "source": "urchade_synthetic_pii_ner_mistral",
        "privacy_mask": privacy_mask
    }
    urchade_converted.append(converted)

# Save
output_path = os.path.join(OUTPUT_DIR, "urchade_unified.json")
with open(output_path, "w") as f:
    json.dump(urchade_converted, f, indent=2)
print(f"   Saved {len(urchade_converted)} samples to {output_path}")

# Show sample
print("   Sample:")
print(f"   Text: {urchade_converted[0]['source_text'][:100]}...")
print(f"   Entities: {urchade_converted[0]['privacy_mask'][:3]}")


# ============================================================
# 3. Convert fewnerd_train.json
#    Format: tokens[], ner_tags[] (numeric)
#    Tags: 0=O, 1=art, 2=building, 3=event, 4=location,
#          5=organization, 6=other, 7=person, 8=product
#    NOTE: This is general NER, not PII-specific
# ============================================================
print("\n" + "=" * 60)
print("3. Converting fewnerd_train.json...")
print("=" * 60)

# Tag mapping for Few-NERD coarse tags
FEWNERD_TAGS = {
    0: "O",
    1: "art",
    2: "building",
    3: "event",
    4: "location",
    5: "organization",
    6: "other",
    7: "person",
    8: "product"
}

with open(os.path.join(INPUT_DIR, "fewnerd_train.json")) as f:
    fewnerd_data = json.load(f)

print(f"   Loaded {len(fewnerd_data)} samples")

fewnerd_converted = []

for item in fewnerd_data:
    tokens = item.get("tokens", [])
    ner_tags = item.get("ner_tags", [])

    # Reconstruct text from tokens
    text = " ".join(tokens)

    # Find entity spans
    privacy_mask = []
    current_entity_tokens = []
    current_label = None
    current_start = 0
    char_pos = 0

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        label = FEWNERD_TAGS.get(tag, "O")

        if label != "O":
            if current_label is None:
                # Start new entity
                current_label = label
                current_start = char_pos
                current_entity_tokens = [token]
            elif label == current_label:
                # Continue same entity
                current_entity_tokens.append(token)
            else:
                # Different entity - save previous and start new
                entity_text = " ".join(current_entity_tokens)
                privacy_mask.append({
                    "label": current_label.upper(),
                    "start": current_start,
                    "end": current_start + len(entity_text),
                    "value": entity_text
                })
                current_label = label
                current_start = char_pos
                current_entity_tokens = [token]
        else:
            if current_label is not None:
                # End of entity
                entity_text = " ".join(current_entity_tokens)
                privacy_mask.append({
                    "label": current_label.upper(),
                    "start": current_start,
                    "end": current_start + len(entity_text),
                    "value": entity_text
                })
                current_label = None
                current_entity_tokens = []

        char_pos += len(token) + 1  # +1 for space

    # Handle entity at end of sequence
    if current_label is not None:
        entity_text = " ".join(current_entity_tokens)
        privacy_mask.append({
            "label": current_label.upper(),
            "start": current_start,
            "end": current_start + len(entity_text),
            "value": entity_text
        })

    converted = {
        "source_text": text,
        "language": "en",
        "source": "fewnerd",
        "privacy_mask": privacy_mask
    }
    fewnerd_converted.append(converted)

# Save
output_path = os.path.join(OUTPUT_DIR, "fewnerd_unified.json")
with open(output_path, "w") as f:
    json.dump(fewnerd_converted, f, indent=2)
print(f"   Saved {len(fewnerd_converted)} samples to {output_path}")

# Show sample with entities
for sample in fewnerd_converted[:50]:
    if sample["privacy_mask"]:
        print("   Sample:")
        print(f"   Text: {sample['source_text'][:100]}...")
        print(f"   Entities: {sample['privacy_mask'][:3]}")
        break


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Conversion Complete!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFiles created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.json'):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        with open(path) as file:
            data = json.load(file)
        print(f"  {f:40s} {size_mb:8.2f} MB  ({len(data)} samples)")
