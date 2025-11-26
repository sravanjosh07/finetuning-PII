"""
Relabel Training Data using LLM (File-by-File)

Processes each label file separately from training_by_label/ directory.
Starts with smaller files first for quick feedback.

Features:
- File-by-file processing (safer, resumable)
- Starts with low-count labels first
- Per-file checkpointing
- Parallel processing within each file
- Outputs one clean file per label

Usage:
    # Test first (5 samples per label)
    TEST_MODE=true python utils/relabel_training_data.py

    # Full run
    python utils/relabel_training_data.py

    # More workers (adjust based on your machine)
    NUM_WORKERS=6 python utils/relabel_training_data.py

    # Better quality model (slower)
    OLLAMA_MODEL=qwen2.5:32b NUM_WORKERS=2 python utils/relabel_training_data.py

    # Process specific label only
    LABEL=amount python utils/relabel_training_data.py
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

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"

INPUT_DIR = DATA_DIR / "training_by_label"
OUTPUT_DIR = DATA_DIR / "LLM-relabeled-data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_SAMPLES = int(os.environ.get("TEST_SAMPLES", "5"))
SPECIFIC_LABEL = os.environ.get("LABEL", None)  # Process only this label

REQUEST_TIMEOUT = 120
MAX_RETRIES = 5  # More retries for resilience
CHECKPOINT_EVERY = 50

write_lock = Lock()

# ============================================================================
# 24 SIMPLIFIED LABELS
# ============================================================================

ALLOWED_LABELS = [
    "date", "full name", "username",
    "social security number", "tax identification number",
    "passport number", "driver's license number", "identification number",
    "phone number", "address", "email address", "ip address", "fax number",
    "credit card number", "credit score", "bank account number",
    "amount", "iban", "insurance number",
    "medical condition", "medication", "medical treatment",
    "organization", "url",
]

# ============================================================================
# RELABELING PROMPT
# ============================================================================

SYSTEM_PROMPT = f"""You are a PII annotation expert. Verify and normalize entity annotations.

ALLOWED LABELS (use ONLY these):
{', '.join(ALLOWED_LABELS)}

TASK:
1. Check if ENTITY VALUE exists in TEXT at the given position
2. Normalize LABEL to one of the allowed labels
3. Fix position if needed
4. Reject if invalid (placeholder, field name, code variable, etc.)

LABEL MAPPING:
- Names (firstname, lastname, person, per) → "full name"
- Dates (dob, birthday, date_of_birth) → "date"
- SSN (ssn, social_security) → "social security number"
- Phone (mobile, telephone, cell) → "phone number"
- Email (email, e-mail) → "email address"
- Address (street, city, location, loc, zip) → "address"
- Credit card (card_number, cc) → "credit card number"
- Account (account_number, routing, swift) → "bank account number"
- Money (balance, salary, transaction, financial) → "amount"
- Company (org, company, employer) → "organization"
- IDs (student_id, employee_id, national_id) → "identification number"
- Insurance (health_insurance, policy) → "insurance number"
- URL (website, link, uri) → "url"

REJECT IF:
- Field names: credit_card_number, user_email, firstName (snake_case/camelCase)
- Placeholders: XXXXXXXXXXXX, [REDACTED], N/A, TBD
- Empty/whitespace values
- Value doesn't exist in text
- Generic non-PII text

OUTPUT (JSON only):
Valid: {{"status": "valid", "label": "<label>", "value": "<text>", "start": <int>, "end": <int>}}
Fixed: {{"status": "fixed", "label": "<label>", "value": "<text>", "start": <int>, "end": <int>}}
Rejected: {{"status": "rejected", "reason": "<brief>"}}
"""


def create_prompt(text, entity):
    """Create prompt for single entity"""
    # Truncate long texts but keep entity context
    max_len = 600
    if len(text) > max_len:
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        ctx_start = max(0, start - 150)
        ctx_end = min(len(text), end + 150)
        text = text[ctx_start:ctx_end]
        entity = entity.copy()
        entity["start"] = start - ctx_start
        entity["end"] = end - ctx_start

    return f'''TEXT: "{text}"
ENTITY: value="{entity.get('value', '')}", label="{entity.get('label', '')}", start={entity.get('start', 0)}, end={entity.get('end', 0)}
OUTPUT:'''


def extract_json(text):
    """Extract JSON from response"""
    text = text.strip()
    # Remove markdown
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Find JSON
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON found")


def call_llm(text, entity):
    """Call LLM to relabel entity - with robust error handling"""
    prompt = create_prompt(text, entity)

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                },
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()

            content = r.json().get("response", "")
            result = extract_json(content)

            status = result.get("status", "rejected")

            if status in ["valid", "fixed"]:
                label = result.get("label", "")
                if label not in ALLOWED_LABELS:
                    return "rejected", {"reason": f"invalid label: {label}"}
                return status, {
                    "label": label,
                    "value": result.get("value", entity.get("value", "")),
                    "start": result.get("start", entity.get("start", 0)),
                    "end": result.get("end", entity.get("end", 0)),
                }
            else:
                return "rejected", {"reason": result.get("reason", "unknown")}

        except requests.exceptions.Timeout:
            # Timeout - wait longer before retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return "error", {"reason": "timeout"}

        except requests.exceptions.ConnectionError:
            # Ollama might be down - wait and retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(30)  # Wait 30s for Ollama to recover
            else:
                return "error", {"reason": "connection_error"}

        except requests.exceptions.HTTPError as e:
            # Server error - might be overloaded
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return "error", {"reason": f"http_{e.response.status_code}"}

        except ValueError as e:
            # JSON parse error - bad response, retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return "error", {"reason": "parse_error"}

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return "error", {"reason": str(e)[:50]}

    return "error", {"reason": "max retries"}


def process_sample(sample):
    """Process single sample - verify primary entity, keep ALL entities"""
    text = sample.get("source_text", "")
    entities = sample.get("privacy_mask", [])

    if not entities or not text:
        return "rejected", None, "empty"

    # Verify only the PRIMARY (first) entity
    primary_entity = entities[0]
    status, result = call_llm(text, primary_entity)

    if status in ["valid", "fixed"]:
        # Keep ALL entities, but update primary if it was fixed
        updated_entities = []
        for i, ent in enumerate(entities):
            if i == 0:
                # Update primary entity with verified/fixed values
                updated_entities.append({
                    "label": result["label"],
                    "start": result["start"],
                    "end": result["end"],
                    "value": result["value"],
                })
            else:
                # Keep other entities as-is (remove original_label if present)
                updated_entities.append({
                    "label": ent.get("label", ""),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "value": ent.get("value", ""),
                })

        return status, {
            "source_text": text,
            "language": sample.get("language", "en"),
            "source": sample.get("source", "unknown"),
            "privacy_mask": updated_entities
        }, None
    else:
        return status, sample, result.get("reason", "unknown")


def load_checkpoint(label_name):
    """Load checkpoint for a label"""
    checkpoint_file = CHECKPOINT_DIR / f"{label_name}_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"processed": 0, "valid": [], "fixed": [], "rejected": [], "errors": []}


def save_checkpoint(label_name, checkpoint):
    """Save checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{label_name}_checkpoint.json"
    with write_lock:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)


def process_file(input_file):
    """Process a single label file"""
    label_name = input_file.stem  # filename without extension

    # Load input
    with open(input_file) as f:
        samples = json.load(f)

    # Load checkpoint
    checkpoint = load_checkpoint(label_name)
    start_idx = checkpoint["processed"]

    if start_idx >= len(samples):
        valid_count = len(checkpoint["valid"]) + len(checkpoint["fixed"])
        return checkpoint, f"Already done: {valid_count} valid"

    remaining = samples[start_idx:]
    if TEST_MODE:
        remaining = remaining[:TEST_SAMPLES]

    results = {
        "valid": list(checkpoint["valid"]),
        "fixed": list(checkpoint["fixed"]),
        "rejected": list(checkpoint["rejected"]),
        "errors": list(checkpoint["errors"]),
    }

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_sample, s): i for i, s in enumerate(remaining)}

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {label_name[:25]:25s}",
            leave=True,
            ncols=90
        )

        for future in pbar:
            idx = futures[future]
            try:
                status, output, reason = future.result()

                if status == "valid":
                    results["valid"].append(output)
                elif status == "fixed":
                    results["fixed"].append(output)
                elif status == "rejected":
                    results["rejected"].append({"reason": reason})
                else:
                    results["errors"].append({"reason": reason})

            except Exception as e:
                results["errors"].append({"reason": str(e)[:50]})

            # Checkpoint periodically
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                checkpoint = {"processed": start_idx + idx + 1, **results}
                save_checkpoint(label_name, checkpoint)

            pbar.set_postfix({
                "✓": len(results["valid"]) + len(results["fixed"]),
                "✗": len(results["rejected"]),
            })

    # Final checkpoint
    checkpoint = {"processed": start_idx + len(remaining), **results}
    save_checkpoint(label_name, checkpoint)

    return checkpoint, None


def main():
    print("=" * 70)
    print("RELABEL TRAINING DATA (File-by-File)")
    print("=" * 70)

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        print(f"\n✓ Ollama: {OLLAMA_URL}")
        print(f"  Model: {MODEL}")
        print(f"  Workers: {NUM_WORKERS}")
    except Exception as e:
        print(f"\n✗ Ollama not available: {e}")
        print("  Run: ollama serve")
        return

    # Find input files
    print(f"\n{'='*70}")
    print("INPUT FILES")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"✗ Input directory not found: {INPUT_DIR}")
        return

    input_files = sorted(INPUT_DIR.glob("*.json"))

    if not input_files:
        print(f"✗ No JSON files in {INPUT_DIR}")
        return

    # Get file sizes and sort by sample count (ascending)
    file_info = []
    for f in input_files:
        with open(f) as fp:
            count = len(json.load(fp))
        file_info.append((f, count))

    # Sort by count (smallest first)
    file_info.sort(key=lambda x: x[1])

    # Filter specific label if requested
    if SPECIFIC_LABEL:
        file_info = [(f, c) for f, c in file_info if SPECIFIC_LABEL in f.stem]
        if not file_info:
            print(f"✗ No file matching label: {SPECIFIC_LABEL}")
            return

    print(f"\nFound {len(file_info)} files (sorted by size, smallest first):\n")
    total_samples = 0
    for f, count in file_info:
        total_samples += count
        print(f"  {f.stem:35s} {count:>8,} samples")

    print(f"\n  {'TOTAL':35s} {total_samples:>8,} samples")

    # Time estimate
    samples_per_min = NUM_WORKERS * 10
    est_hours = total_samples / samples_per_min / 60
    print(f"\nEstimated time: {est_hours:.1f} hours (~{samples_per_min} samples/min)")

    if TEST_MODE:
        print(f"\n⚠ TEST MODE: {TEST_SAMPLES} samples per file")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each file
    print(f"\n{'='*70}")
    print("PROCESSING")
    print("=" * 70)

    overall_stats = {
        "valid": 0, "fixed": 0, "rejected": 0, "errors": 0
    }

    for input_file, sample_count in file_info:
        label_name = input_file.stem
        print(f"\n{label_name} ({sample_count:,} samples)")

        checkpoint, skip_msg = process_file(input_file)

        if skip_msg:
            print(f"  {skip_msg}")

        v = len(checkpoint["valid"])
        f = len(checkpoint["fixed"])
        r = len(checkpoint["rejected"])
        e = len(checkpoint["errors"])

        overall_stats["valid"] += v
        overall_stats["fixed"] += f
        overall_stats["rejected"] += r
        overall_stats["errors"] += e

        # Save output file for this label
        clean_data = checkpoint["valid"] + checkpoint["fixed"]
        if clean_data:
            output_file = OUTPUT_DIR / f"{label_name}.json"
            with open(output_file, "w") as fp:
                json.dump(clean_data, fp)

            size_kb = os.path.getsize(output_file) / 1024
            print(f"  → Saved {len(clean_data):,} samples to {label_name}.json ({size_kb:.0f} KB)")

        print(f"  Valid: {v:,}  Fixed: {f:,}  Rejected: {r:,}  Errors: {e:,}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    total_out = overall_stats["valid"] + overall_stats["fixed"]
    print(f"\n  Input:    {total_samples:>10,}")
    print(f"  Valid:    {overall_stats['valid']:>10,}")
    print(f"  Fixed:    {overall_stats['fixed']:>10,}")
    print(f"  Rejected: {overall_stats['rejected']:>10,}")
    print(f"  Errors:   {overall_stats['errors']:>10,}")
    print(f"  ─────────────────────────")
    print(f"  Output:   {total_out:>10,} ({total_out/total_samples*100:.1f}%)")

    print(f"\n✓ Output directory: {OUTPUT_DIR}")
    print(f"  Files: {len(list(OUTPUT_DIR.glob('*.json')))} label files")
    print(f"\n  (Combine files manually after completion if needed)")


if __name__ == "__main__":
    main()
