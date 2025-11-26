"""
Verify Training Data using LLM

Verifies the combined training dataset by checking if:
1. The entity value actually exists in the text at the given position
2. The label is appropriate for the entity
3. The entity is actually PII (not a field name, placeholder, etc.)

Optimized for long-running jobs:
- Parallel processing with configurable workers
- Checkpoint/resume support (saves progress per label)
- Processes label-by-label (get usable data even if interrupted)
- Simple accept/reject verification (faster than relabeling)

Usage:
    # Full run with 4 workers
    python utils/verify_training_data.py

    # Test mode (10 samples)
    TEST_MODE=true python utils/verify_training_data.py

    # Use different model
    OLLAMA_MODEL=qwen2.5:14b python utils/verify_training_data.py

    # Adjust parallelism
    NUM_WORKERS=8 python utils/verify_training_data.py
"""

import json
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import requests
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

# Model (qwen2.5:14b is good balance of speed/quality for verification)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Parallelism (adjust based on your machine)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"

INPUT_FILE = DATA_DIR / "combined_training_24labels.json"
OUTPUT_DIR = DATA_DIR / "verified_training"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Test mode
TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_SAMPLES = int(os.environ.get("TEST_SAMPLES", "10"))

# Performance
REQUEST_TIMEOUT = 60  # Shorter timeout for simple verification
MAX_RETRIES = 2
CHECKPOINT_EVERY = 100  # Save checkpoint every N samples

# File lock for thread-safe writing
write_lock = Lock()

# ============================================================================
# VERIFICATION PROMPT (Simple accept/reject - much faster than relabeling)
# ============================================================================

SYSTEM_PROMPT = """You are a PII data quality checker. Your task is to verify if a labeled entity is correct.

Given:
- TEXT: The source text
- ENTITY: The extracted entity value
- LABEL: The assigned label
- POSITION: Start and end character positions

Verify:
1. Does the entity text exist at the given position in TEXT?
2. Is the LABEL appropriate for this entity?
3. Is this actually PII (not a field name, placeholder, code variable, or generic text)?

RULES FOR REJECTION:
- Field names / column headers (e.g., "credit_card_number", "user_email", "firstName")
- Placeholder text (e.g., "XXXXXXXXXXXX", "[REDACTED]", "N/A")
- Code variables / technical identifiers (snake_case, camelCase)
- Generic non-PII text incorrectly labeled
- Position mismatch (entity not at stated position)
- Wrong label for the entity type

RULES FOR ACCEPTANCE:
- Real PII values at correct positions with appropriate labels
- Actual names, emails, phone numbers, addresses, IDs, etc.
- Financial data (account numbers, amounts, IBANs)
- Medical information (conditions, medications)

OUTPUT: Return ONLY a JSON object:
{"valid": true} or {"valid": false, "reason": "brief reason"}

Examples:

TEXT: "Contact John Smith at john@email.com"
ENTITY: "John Smith", LABEL: "full name", START: 8, END: 18
OUTPUT: {"valid": true}

TEXT: "Fields: first_name, last_name, email_address"
ENTITY: "first_name", LABEL: "full name", START: 8, END: 18
OUTPUT: {"valid": false, "reason": "field name, not actual PII"}

TEXT: "SSN: XXXXXXXXXXXX"
ENTITY: "XXXXXXXXXXXX", LABEL: "social security number", START: 5, END: 17
OUTPUT: {"valid": false, "reason": "placeholder text"}

TEXT: "Balance: $5,000.00 in account 12345678"
ENTITY: "$5,000.00", LABEL: "amount", START: 9, END: 18
OUTPUT: {"valid": true}
"""


def create_verification_prompt(text, entity):
    """Create a simple verification prompt"""
    return f'''TEXT: "{text[:500]}"
ENTITY: "{entity.get('value', '')}"
LABEL: "{entity.get('label', '')}"
START: {entity.get('start', 0)}, END: {entity.get('end', 0)}
OUTPUT:'''


def extract_json(text):
    """Extract JSON from LLM response"""
    text = text.strip()

    # Remove markdown
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Find JSON object
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    return {"valid": False, "reason": "parse error"}


def verify_entity(text, entity):
    """Call LLM to verify a single entity"""
    prompt = create_verification_prompt(text, entity)

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 100,  # Short response
                    }
                },
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()

            content = r.json().get("response", "")
            result = extract_json(content)
            return result.get("valid", False), result.get("reason", "")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return False, f"error: {str(e)[:50]}"

    return False, "max retries"


def verify_sample(sample):
    """Verify a single sample (text + entity)"""
    text = sample.get("source_text", "")
    entities = sample.get("privacy_mask", [])

    if not entities:
        return None, "no entities"

    entity = entities[0]  # Each sample has 1 entity in our format

    # Quick pre-check: verify entity exists at position
    start = entity.get("start", 0)
    end = entity.get("end", 0)
    value = entity.get("value", "")

    if start < len(text) and end <= len(text):
        actual_text = text[start:end]
        if actual_text != value:
            # Position mismatch - might still be valid if text exists elsewhere
            if value not in text:
                return None, "entity not in text"

    # LLM verification
    valid, reason = verify_entity(text, entity)

    if valid:
        return sample, None
    else:
        return None, reason


def load_checkpoint(label):
    """Load checkpoint for a label"""
    checkpoint_file = CHECKPOINT_DIR / f"{label}_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"processed": 0, "valid": [], "invalid": []}


def save_checkpoint(label, checkpoint):
    """Save checkpoint for a label"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{label}_checkpoint.json"
    with write_lock:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)


def process_label(label, samples):
    """Process all samples for a single label"""
    # Load checkpoint
    checkpoint = load_checkpoint(label)
    start_idx = checkpoint["processed"]

    if start_idx >= len(samples):
        print(f"  {label}: Already complete ({len(checkpoint['valid'])} valid)")
        return checkpoint["valid"], checkpoint["invalid"]

    remaining = samples[start_idx:]

    if TEST_MODE:
        remaining = remaining[:TEST_SAMPLES]

    valid_samples = checkpoint["valid"]
    invalid_samples = checkpoint["invalid"]

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(verify_sample, s): i for i, s in enumerate(remaining)}

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {label[:20]:20s}",
            leave=False
        )

        for future in pbar:
            idx = futures[future]
            try:
                result, reason = future.result()
                if result:
                    valid_samples.append(result)
                else:
                    invalid_samples.append({
                        "sample": remaining[idx],
                        "reason": reason
                    })
            except Exception as e:
                invalid_samples.append({
                    "sample": remaining[idx],
                    "reason": str(e)[:50]
                })

            # Checkpoint periodically
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                checkpoint = {
                    "processed": start_idx + idx + 1,
                    "valid": valid_samples,
                    "invalid": invalid_samples
                }
                save_checkpoint(label, checkpoint)
                pbar.set_postfix({"valid": len(valid_samples), "invalid": len(invalid_samples)})

    # Final checkpoint
    checkpoint = {
        "processed": start_idx + len(remaining),
        "valid": valid_samples,
        "invalid": invalid_samples
    }
    save_checkpoint(label, checkpoint)

    return valid_samples, invalid_samples


def main():
    print("=" * 70)
    print("TRAINING DATA VERIFICATION")
    print("=" * 70)

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"\n✓ Ollama connected: {OLLAMA_URL}")
        print(f"  Model: {MODEL}")
        print(f"  Workers: {NUM_WORKERS}")
    except Exception as e:
        print(f"\n✗ Cannot connect to Ollama: {e}")
        print("  Run: ollama serve")
        return

    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    with open(INPUT_FILE) as f:
        all_data = json.load(f)

    print(f"✓ Loaded {len(all_data):,} samples from {INPUT_FILE.name}")

    # Group by label
    samples_by_label = {}
    for sample in all_data:
        for entity in sample.get("privacy_mask", []):
            label = entity.get("label", "unknown")
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(sample)
            break  # One entity per sample

    print(f"\nLabels to process: {len(samples_by_label)}")
    for label, samples in sorted(samples_by_label.items(), key=lambda x: -len(x[1])):
        print(f"  {label:30s} {len(samples):>8,}")

    # Estimate time
    total_samples = sum(len(s) for s in samples_by_label.values())
    samples_per_min = NUM_WORKERS * 20  # ~20 verifications/min per worker
    est_minutes = total_samples / samples_per_min

    print(f"\nEstimated time: {est_minutes/60:.1f} hours ({est_minutes:.0f} min)")
    print(f"  Rate: ~{samples_per_min} samples/min with {NUM_WORKERS} workers")

    if TEST_MODE:
        print(f"\n⚠ TEST MODE: {TEST_SAMPLES} samples per label")

    # Process each label
    print(f"\n{'='*70}")
    print("VERIFYING")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_valid = []
    all_invalid = []

    for label in sorted(samples_by_label.keys()):
        samples = samples_by_label[label]
        print(f"\n{label} ({len(samples):,} samples)")

        valid, invalid = process_label(label, samples)
        all_valid.extend(valid)
        all_invalid.extend(invalid)

        print(f"  ✓ Valid: {len(valid):,}  ✗ Invalid: {len(invalid):,}")

    # Save final outputs
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("=" * 70)

    # Save verified dataset
    output_file = OUTPUT_DIR / "verified_training_data.json"
    with open(output_file, "w") as f:
        json.dump(all_valid, f)
    print(f"✓ Verified data: {output_file}")
    print(f"  {len(all_valid):,} samples ({os.path.getsize(output_file)/1024/1024:.1f} MB)")

    # Save rejected samples (for review)
    rejected_file = OUTPUT_DIR / "rejected_samples.json"
    with open(rejected_file, "w") as f:
        json.dump(all_invalid, f, indent=2)
    print(f"✓ Rejected data: {rejected_file}")
    print(f"  {len(all_invalid):,} samples")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    total = len(all_valid) + len(all_invalid)
    valid_pct = len(all_valid) / total * 100 if total > 0 else 0
    print(f"  Total processed: {total:,}")
    print(f"  Valid:           {len(all_valid):,} ({valid_pct:.1f}%)")
    print(f"  Rejected:        {len(all_invalid):,} ({100-valid_pct:.1f}%)")


if __name__ == "__main__":
    main()
