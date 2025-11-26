"""
Verify Negative Examples (PII-free data)

This script verifies that data samples are free from all PII labels.
These verified samples will be used as negative examples for training.

Optimized for Mac Mini (48GB RAM) with qwen2.5:32b

Usage:
  # Full run
  python utils/verify_negative_examples.py

  # Test mode (5 samples)
  TEST_MODE=true python utils/verify_negative_examples.py

  # Use faster 14b model
  OLLAMA_MODEL=qwen2.5:14b python utils/verify_negative_examples.py

  # Resume after interruption
  python utils/verify_negative_examples.py
"""
import json
import re
import time
import os
from pathlib import Path
from tqdm import tqdm
import requests

# ============================================================================
# CONFIG
# ============================================================================

# Model settings
MODEL = "qwen2.5:32b"  # Use "qwen2.5:14b" for faster processing
OLLAMA_URL = "http://localhost:11434"

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_FILE = str(PROJECT_DIR / "data" / "unlabeled" / "sampled_500_negative.ndjson")
OUTPUT_FILE = str(PROJECT_DIR / "data" / "verified_negative_examples.ndjson")
REJECTED_FILE = str(PROJECT_DIR / "data" / "rejected_negative_examples.ndjson")

# Test mode - uncomment to test with 5 samples
TEST_MODE = False
TEST_NUM_SAMPLES = None

# Full run - uncomment to process all samples
# TEST_MODE = False
# TEST_NUM_SAMPLES = None

REQUEST_TIMEOUT = 180
MAX_RETRIES = 3

# ============================================================================
# 27 PII LABELS - Check for all of these
# ============================================================================
LABELS = [
    "date",
    "full name",
    "social security number",
    "tax identification number",
    "drivers license number",
    "identity card number",
    "passport number",
    "birth certificate number",
    "student id number",
    "phone number",
    "mobile phone number",
    "fax number",
    "email address",
    "ip address",
    "address",
    "credit card number",
    "credit score",
    "bank account number",
    "amount",
    "iban",
    "health insurance id number",
    "insurance plan number",
    "national health insurance number",
    "medical condition",
    "medication",
    "medical treatment",
    "username",
    "organization",
]

# ============================================================================
# PROMPT
# ============================================================================
SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) detector.

TASK: Analyze the given text and determine if it contains ANY PII entities.

ALLOWED_PII_LABELS (check for these 27 types):
- date: Any date including birth dates, event dates, document dates
- full name: Complete person name or first/last name
- social security number: US SSN format like "123-45-6789"
- tax identification number: Tax IDs like EIN, TIN, ITIN
- drivers license number: State/country issued driver's license numbers
- identity card number: National ID cards, government IDs
- passport number: Passport numbers from any country
- birth certificate number: Birth certificate registration numbers
- student id number: School/university student IDs
- phone number: Landline telephone numbers with area code
- mobile phone number: Cell/mobile phone numbers
- fax number: Fax machine numbers
- email address: Email addresses (e.g., "user@domain.com")
- ip address: IPv4 or IPv6 addresses
- address: Physical/mailing addresses including street, city, state, zip
- credit card number: 13-19 digit card numbers
- credit score: Numeric credit scores
- bank account number: Bank account numbers
- amount: Sensitive financial amounts (account balances, salaries, loan amounts). NOT generic prices.
- iban: International Bank Account Numbers
- health insurance id number: Health insurance member/policy ID numbers
- insurance plan number: Insurance plan identifiers
- national health insurance number: Government health insurance IDs
- medical condition: Diseases, diagnoses, health conditions
- medication: Drug names, prescriptions
- medical treatment: Medical procedures, treatments, therapies
- username: Online usernames, login IDs, handles
- organization: Company names, institution names

STRICT RULES:
1. Scan the text carefully for ANY of the 27 PII types listed above
2. DO NOT label these as PII:
   - Job titles, ages, ticket numbers, reference codes
   - Field names, variable names (e.g., "credit_card_number", "user_id")
   - Generic prices or list prices
   - Placeholder text like "XXXXXXXXXXXX"
   - Code identifiers (snake_case or camelCase)
3. If you find ANY real PII, list all entities found
4. If the text is completely free of PII, return empty list

EXAMPLES:

Text: "The product costs $29.99 and ships in 3-5 business days."
Output: {"has_pii": false, "entities": [], "reason": "No PII found - only generic pricing and shipping info"}

Text: "Contact support at support@company.com for assistance."
Output: {"has_pii": true, "entities": [{"text": "support@company.com", "label": "email address", "start": 18, "end": 37}], "reason": "Contains email address"}

Text: "John Smith called about his account balance of $5,000."
Output: {"has_pii": true, "entities": [{"text": "John Smith", "label": "full name", "start": 0, "end": 10}, {"text": "$5,000", "label": "amount", "start": 48, "end": 54}], "reason": "Contains person name and financial amount"}

Text: "The average temperature was 72 degrees on Tuesday."
Output: {"has_pii": false, "entities": [], "reason": "No PII found - weather data is not personally identifiable"}

Text: "Update the user_id and email_address fields in the database."
Output: {"has_pii": false, "entities": [], "reason": "No PII found - these are field names, not actual data"}

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "has_pii": true/false,
  "entities": [...],
  "reason": "brief explanation"
}

If has_pii is true, include all found entities with: "text", "label", "start", "end"
If has_pii is false, entities should be empty list []"""


def create_prompt(text):
    """Create prompt for LLM to check for PII"""
    return f'Text: "{text}"\nOutput:'


def extract_json(text):
    """Extract JSON from LLM response"""
    text = text.strip()

    # Remove markdown code blocks
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Extract JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    return json.loads(text)


def load_progress():
    """Load already processed sample indices"""
    processed = set()

    for filepath in [OUTPUT_FILE, REJECTED_FILE]:
        if not os.path.exists(filepath):
            continue

        with open(filepath) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip().rstrip(','))
                        processed.add(data["sample_idx"])
                    except Exception as e:
                        print(f"Warning: Could not parse line: {e}")
                        continue

    return processed


def call_ollama(text):
    """Call Ollama API to check for PII"""
    prompt = create_prompt(text)

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
                        "num_predict": 4096,
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                },
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()

            content = r.json().get("response", "")
            if not content:
                raise ValueError("Empty response from model")

            result = extract_json(content)
            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\n  Retry {attempt+1}/{MAX_RETRIES}: {str(e)[:60]}")
                time.sleep(2 ** attempt)
            else:
                print(f"\n  Failed after {MAX_RETRIES} attempts: {str(e)[:60]}")
                return {"has_pii": None, "entities": [], "reason": f"Error: {str(e)[:100]}"}

    return {"has_pii": None, "entities": [], "reason": "Max retries exceeded"}


def load_data(filepath):
    """Load JSON or NDJSON file"""
    with open(filepath, 'r') as f:
        # Try loading as regular JSON first
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
        except:
            pass

    # Fall back to NDJSON
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            if line:
                data.append(json.loads(line))
    return data


def main():
    print("=" * 80)
    print("VERIFY NEGATIVE EXAMPLES (PII-FREE DATA)")
    print("=" * 80)

    # Check Ollama connection
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"\n✓ Connected to Ollama: {OLLAMA_URL}")
        print(f"  Available models: {', '.join(models)}")

        if not any(MODEL in m for m in models):
            print(f"\n✗ Model {MODEL} not found!")
            print(f"  Run: ollama pull {MODEL}")
            return

        print(f"  Using model: {MODEL}")

    except Exception as e:
        print(f"\n✗ Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"  Error: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        return

    # Load input data
    print(f"\n{'=' * 80}")
    print("LOADING DATA")
    print('=' * 80)
    print(f"Input: {INPUT_FILE}")

    try:
        data = load_data(INPUT_FILE)
        print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        print(f"✗ Failed to load input file: {e}")
        print(f"\nNote: Update INPUT_FILE path to point to your negative candidate data")
        return

    # Check progress
    processed = load_progress()
    remaining = [i for i in range(len(data)) if i not in processed]

    print(f"\nProgress:")
    print(f"  Total samples:     {len(data)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  Remaining:         {len(remaining)}")

    if TEST_MODE:
        remaining = remaining[:TEST_NUM_SAMPLES]
        print(f"\n⚠ TEST MODE: Processing only {len(remaining)} samples")

    if not remaining:
        print("\n✓ All samples already processed!")
        print(f"  Verified (no PII): {OUTPUT_FILE}")
        print(f"  Rejected (has PII): {REJECTED_FILE}")
        return

    # Estimate time
    samples_per_minute = 3 if "32b" in MODEL else 5
    estimated_minutes = len(remaining) / samples_per_minute
    estimated_hours = estimated_minutes / 60

    print(f"\nEstimated time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"  Rate: ~{samples_per_minute} samples/minute")

    # Create output directories
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(REJECTED_FILE), exist_ok=True)

    # Process samples
    print(f"\n{'=' * 80}")
    print("PROCESSING")
    print('=' * 80)

    start_time = time.time()
    verified_count = 0  # No PII found - good for negative examples
    rejected_count = 0  # PII found - reject
    error_count = 0     # Could not verify

    for idx in tqdm(remaining, desc="Verifying", unit="sample"):
        sample = data[idx]
        text = sample.get("text", "")

        # Check for PII
        result = call_ollama(text)

        has_pii = result.get("has_pii")
        entities = result.get("entities", [])
        reason = result.get("reason", "")

        # Prepare output
        output_data = {
            "sample_idx": idx,
            "text": text,
            "has_pii": has_pii,
            "found_entities": entities,
            "reason": reason,
        }

        # Save to appropriate file
        if has_pii is None:
            # Error case
            error_count += 1
            with open(REJECTED_FILE, "a") as f:
                f.write(json.dumps(output_data) + "\n")
        elif has_pii:
            # Contains PII - reject
            rejected_count += 1
            with open(REJECTED_FILE, "a") as f:
                f.write(json.dumps(output_data) + "\n")
        else:
            # No PII - verified negative example
            verified_count += 1
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(output_data) + "\n")

    # Summary
    elapsed = time.time() - start_time
    rate = len(remaining) / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print('=' * 80)
    print(f"✓ Processed: {len(remaining)} samples")
    print(f"  Verified (no PII): {verified_count}")
    print(f"  Rejected (has PII): {rejected_count}")
    print(f"  Errors:            {error_count}")
    print(f"  Time:              {elapsed/60:.1f} minutes ({rate*60:.1f} samples/minute)")
    print(f"\nOutputs:")
    print(f"  Clean data: {OUTPUT_FILE}")
    print(f"  Rejected:   {REJECTED_FILE}")


if __name__ == "__main__":
    main()
