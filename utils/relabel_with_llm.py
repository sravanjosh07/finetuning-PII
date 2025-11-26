"""
Re-label Gold Test Dataset using Ollama LLM

This script fixes ground truth annotation issues by:
1. Filling in missing PII entities (samples with 0 entities that should have some)
2. Normalizing labels to the 27 standard labels
3. Merging adjacent name entities (first_name + last_name -> full name)
4. Removing invalid labels (job titles, ticket numbers, etc.)

Optimized for Mac Mini (48GB RAM) with qwen2.5:32b

Usage:
  # Full run (recommended for weekend)
  python utils/relabel_with_llm.py

  # Test mode (5 samples)
  TEST_MODE=true python utils/relabel_with_llm.py

  # Use faster 14b model instead of 32b
  OLLAMA_MODEL=qwen2.5:14b python utils/relabel_with_llm.py

  # Resume after interruption (automatically detects progress)
  python utils/relabel_with_llm.py
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

# Model selection (qwen2.5:32b is best for quality, 14b for speed)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "Data"

INPUT_FILE = str(DATA_DIR / "gold_testdataset_27labels.ndjson")
OUTPUT_FILE = str(PROJECT_DIR / "data" / "gold_testdataset_llm_relabeled.ndjson")

# Test mode (set TEST_MODE=true to test with 5 samples)
TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_NUM_SAMPLES = int(os.environ.get("TEST_NUM_SAMPLES", "5"))

# Performance settings for Mac Mini 48GB
BATCH_SIZE = 1  # Process one at a time for stability
REQUEST_TIMEOUT = 180  # 3 minutes timeout for complex samples
MAX_RETRIES = 3

# ============================================================================
# 27 ALLOWED LABELS - Do not add or invent new labels
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
SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) entity annotator and normalizer.

TASK: Given text and existing entity annotations (which may be incomplete or missing), identify ALL PII entities and normalize labels to match the ALLOWED_LABELS list exactly.

ALLOWED_LABELS (use ONLY these 27 labels, do NOT invent new ones):
- date: Any date including birth dates, event dates, document dates (e.g., "Jan 5, 1990", "2024-01-15", "05/01/1990")
- full name: Complete person name. IMPORTANT: Merge separate first_name + last_name into ONE "full name" entity
- social security number: US SSN format like "123-45-6789" or "123456789"
- tax identification number: Tax IDs like EIN, TIN, ITIN (e.g., "12-3456789")
- drivers license number: State/country issued driver's license numbers
- identity card number: National ID cards, government IDs (NOT SSN, NOT driver's license)
- passport number: Passport numbers from any country
- birth certificate number: Birth certificate registration numbers
- student id number: School/university student IDs
- phone number: Landline telephone numbers with area code
- mobile phone number: Cell/mobile phone numbers
- fax number: Fax machine numbers
- email address: Email addresses (e.g., "user@domain.com")
- ip address: IPv4 (e.g., "192.168.1.1") or IPv6 addresses
- address: Physical/mailing addresses including street, city, state, zip
- credit card number: 13-19 digit card numbers, may include spaces or dashes
- credit score: Numeric credit scores (typically 300-850 range)
- bank account number: Bank account numbers (NOT routing numbers)
- amount: Sensitive financial amounts (account balances, salaries, loan amounts, transaction totals). DO NOT label every price or generic number - only amounts that reveal personal financial information.
- iban: International Bank Account Numbers (starts with country code like "DE89...")
- health insurance id number: Health insurance member/policy ID numbers
- insurance plan number: Insurance plan identifiers
- national health insurance number: Government health insurance IDs (NHS numbers, Medicare numbers)
- medical condition: Diseases, diagnoses, health conditions (e.g., "diabetes", "hypertension")
- medication: Drug names, prescriptions (e.g., "Metformin", "Aspirin")
- medical treatment: Medical procedures, treatments, therapies (e.g., "chemotherapy", "surgery")
- username: Online usernames, login IDs, handles
- organization: Company names, institution names, organization names

STRICT RULES:
1. Use ONLY the 27 labels listed above. Do NOT create new labels.
2. FIND ALL PII in the text, even if not in the original annotations (original may be incomplete or empty)
3. Merge adjacent first_name + last_name into single "full name" with combined text and correct start/end span
4. Map common legacy labels to allowed labels:
   - ssn, socialsecuritynumber -> social security number
   - dateofbirth, date_of_birth, dob -> date
   - creditcardnumber, credit_card -> credit card number
   - taxnum, tin -> tax identification number
   - phone, telephone -> phone number
   - email -> email address
   - ipaddress -> ip address
   - company, org -> organization
   - bank_account_balance, balance, salary -> amount
5. Verify the "text" field matches exactly what appears in the source text
6. Calculate accurate start and end character positions
7. Do NOT label these as PII:
   - Job titles, ages, ticket numbers, reference codes, internal IDs
   - Field names, variable names, column headers (e.g., "credit_card_number", "user_id", "firstName")
   - Code/technical context: snake_case or camelCase identifiers
   - Generic prices, list prices, or public pricing information
   - Generic numbers without context
   - Placeholder text like "XXXXXXXXXXXX"
8. DO label as "amount" only sensitive financial information: account balances, salaries, personal loan amounts, personal transaction totals. Be selective - not every dollar amount is PII.
9. If an entity does not fit any allowed label, do NOT include it
10. DO include medical entities (conditions, medications, treatments) if present in medical contexts

EXAMPLE 1 - Incomplete annotations (original missing entities):
Text: "Contact John Smith at john@email.com or 555-1234"
Original: []
Output: {"normalized_entities":[{"text":"John Smith","label":"full name","start":8,"end":18},{"text":"john@email.com","label":"email address","start":22,"end":36},{"text":"555-1234","label":"phone number","start":40,"end":48}]}

EXAMPLE 2 - Merging names and normalizing:
Text: "Patient John Smith (SSN: 123-45-6789) born 1990-05-15"
Original: [{"text":"John","label":"first_name","start":8,"end":12},{"text":"Smith","label":"last_name","start":13,"end":18},{"text":"123-45-6789","label":"ssn","start":25,"end":36},{"text":"1990-05-15","label":"dateofbirth","start":43,"end":53}]
Output: {"normalized_entities":[{"text":"John Smith","label":"full name","start":8,"end":18},{"text":"123-45-6789","label":"social security number","start":25,"end":36},{"text":"1990-05-15","label":"date","start":43,"end":53}]}

EXAMPLE 3 - Medical context with entities:
Text: "Patient has diabetes and takes Metformin 500mg daily"
Original: []
Output: {"normalized_entities":[{"text":"diabetes","label":"medical condition","start":12,"end":20},{"text":"Metformin","label":"medication","start":31,"end":40}]}

EXAMPLE 4 - Filtering invalid entities and code context:
Text: "Ticket #12345 assigned to Manager role with priority HIGH"
Original: [{"text":"#12345","label":"ticket_id","start":7,"end":13},{"text":"Manager","label":"job_title","start":26,"end":33}]
Output: {"normalized_entities":[]}

EXAMPLE 5 - Code/technical context (field names, NOT actual data):
Text: "Update user profile: credit_card_number, date_of_birth, ipv4_address fields"
Original: [{"text":"credit_card_number","label":"credit_card","start":21,"end":39},{"text":"date_of_birth","label":"date","start":41,"end":54},{"text":"ipv4_address","label":"ip_address","start":56,"end":68}]
Output: {"normalized_entities":[]}

EXAMPLE 6 - Selective amount labeling (personal financial info only):
Text: "Account Balance: $5,000. Salary: $85,000/year. Product price: $49.99"
Original: [{"text":"$5,000","label":"balance","start":17,"end":23},{"text":"$85,000","label":"salary","start":33,"end":40},{"text":"$49.99","label":"price","start":63,"end":69}]
Output: {"normalized_entities":[{"text":"$5,000","label":"amount","start":17,"end":23},{"text":"$85,000","label":"amount","start":33,"end":40}]}

EXAMPLE 7 - Placeholder text (not real PII):
Text: "Policy Number: XXXXXXXXXXXX, Policyholder: Khalid Alali"
Original: [{"text":"XXXXXXXXXXXX","label":"policy_number","start":15,"end":27},{"text":"Khalid Alali","label":"full_name","start":44,"end":56}]
Output: {"normalized_entities":[{"text":"Khalid Alali","label":"full name","start":44,"end":56}]}

OUTPUT FORMAT:
Return ONLY a valid JSON object: {"normalized_entities": [...]}
Each entity must have: "text" (exact string from source), "label" (from allowed list), "start" (character position), "end" (character position)
Do NOT include any explanation, commentary, or markdown formatting."""


def create_prompt(text, entities):
    """Create prompt for LLM with text and existing entities"""
    simple = []
    for e in entities:
        simple.append({
            "text": e.get("text", e.get("entity", "")),
            "label": e.get("label", e.get("types", [""])[0] if "types" in e else ""),
            "start": e.get("start", 0),
            "end": e.get("end", 0)
        })
    return f'Text: "{text}"\nOriginal: {json.dumps(simple)}\nOutput:'


def extract_json(text):
    """Extract JSON from LLM response, handling markdown and other formatting"""
    text = text.strip()

    # Remove markdown code blocks if present
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Extract JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    return json.loads(text)


def load_progress():
    """Load already processed sample indices to support resume"""
    if not os.path.exists(OUTPUT_FILE):
        return set()

    processed = set()
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip().rstrip(','))
                    processed.add(data["sample_idx"])
                except Exception as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue

    return processed


def call_ollama(text, entities):
    """Call Ollama API to normalize entities"""
    prompt = create_prompt(text, entities)

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent output
                        "num_predict": 4096,  # Enough for complex outputs
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

            result = extract_json(content).get("normalized_entities", [])

            # Validate and keep only valid labels
            valid_entities = []
            for e in result:
                if e.get("label") in LABELS:
                    # Ensure all required fields present
                    if all(k in e for k in ["text", "label", "start", "end"]):
                        valid_entities.append(e)

            return valid_entities

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\n  Retry {attempt+1}/{MAX_RETRIES}: {str(e)[:60]}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"\n  Failed after {MAX_RETRIES} attempts: {str(e)[:60]}")
                return []

    return []


def load_ndjson(filepath):
    """Load NDJSON file (handles trailing commas)"""
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
    print("LLM-BASED GROUND TRUTH RELABELING")
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
        data = load_ndjson(INPUT_FILE)
        print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        print(f"✗ Failed to load input file: {e}")
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
        print(f"  Output file: {OUTPUT_FILE}")
        return

    # Estimate time
    samples_per_minute = 3 if "32b" in MODEL else 5  # Rough estimates
    estimated_minutes = len(remaining) / samples_per_minute
    estimated_hours = estimated_minutes / 60

    print(f"\nEstimated time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"  Rate: ~{samples_per_minute} samples/minute")

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Process samples
    print(f"\n{'=' * 80}")
    print("PROCESSING")
    print('=' * 80)

    start_time = time.time()
    success_count = 0
    error_count = 0

    for idx in tqdm(remaining, desc="Relabeling", unit="sample"):
        sample = data[idx]
        text = sample.get("text", "")

        # Get existing entities (may be empty)
        entities_key = 'normalized_entities' if 'normalized_entities' in sample else 'entities'
        original_entities = sample.get(entities_key, [])

        # Call LLM to normalize/complete annotations
        normalized = call_ollama(text, original_entities)

        if normalized or len(original_entities) == 0:
            success_count += 1
        else:
            error_count += 1

        # Save result
        output_data = {
            "sample_idx": idx,
            "text": text,
            "normalized_entities": normalized,
            "original_entities": original_entities,
            "original_entity_count": len(original_entities),
            "normalized_entity_count": len(normalized),
        }

        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(output_data) + "\n")

    # Summary
    elapsed = time.time() - start_time
    rate = len(remaining) / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print('=' * 80)
    print(f"✓ Processed: {len(remaining)} samples")
    print(f"  Successful: {success_count}")
    print(f"  Errors:     {error_count}")
    print(f"  Time:       {elapsed/60:.1f} minutes ({rate*60:.1f} samples/minute)")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total in output file: {len(processed) + len(remaining)} samples")


if __name__ == "__main__":
    main()
