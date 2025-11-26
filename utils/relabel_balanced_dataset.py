"""
Re-label Balanced Dataset using Ollama LLM

Relabels balanced_augmented_test_dataset.json with LLM to:
1. Merge adjacent name entities (first_name + last_name -> full name)
2. Normalize labels to 26 standard labels (mobile phone → phone number)
3. Fix mislabeled entities
4. Add missing PII entities

Writes NDJSON format for easy streaming and resume support.

Usage:
    python utils/relabel_balanced_dataset.py

    # To test on fewer samples, edit TEST_MODE and NUM_TEST_SAMPLES below
    # To resume after interruption, just run again - auto-detects progress
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import relabel functions
from utils.relabel_with_llm import call_ollama, LABELS, MODEL, OLLAMA_URL
import requests

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Test mode - set to True to process only a few samples first
TEST_MODE = False  # Change to True for testing
NUM_TEST_SAMPLES = None  # How many samples to test if TEST_MODE = True

# Input and output files
INPUT_FILE = PROJECT_DIR / "data" / "balanced_augmented_test_dataset.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "balanced_augmented_relabeled.ndjson"


def load_progress():
    """
    Load already processed sample indices to support resume.

    How it works:
    - Reads the NDJSON output file if it exists
    - Extracts sample_idx from each line
    - Returns set of indices already processed
    - If script crashes, running again will skip already-done samples
    """
    # If output file doesn't exist yet, nothing has been processed
    if not OUTPUT_FILE.exists():
        return set()

    processed = set()

    # Read NDJSON file line by line
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    # Parse JSON from this line
                    data = json.loads(line.strip())
                    # Add this sample index to processed set
                    processed.add(data["sample_idx"])
                except Exception as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue

    return processed


def main():
    print("=" * 80)
    print("LLM RELABELING - BALANCED AUGMENTED DATASET")
    print("=" * 80)

    # Check Ollama connection
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"\n✓ Connected to Ollama: {OLLAMA_URL}")
        print(f"  Available models: {', '.join(models[:5])}")

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
    print(f"Input: {INPUT_FILE.name}")

    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        print(f"✗ Failed to load input file: {e}")
        return

    # Check progress (resume support)
    processed = load_progress()
    remaining_indices = [i for i in range(len(data)) if i not in processed]

    print(f"\nProgress:")
    print(f"  Total samples:     {len(data)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  Remaining:         {len(remaining_indices)}")

    # Apply test mode limit if enabled
    if TEST_MODE:
        remaining_indices = remaining_indices[:NUM_TEST_SAMPLES]
        print(f"\n⚠ TEST MODE: Processing only {len(remaining_indices)} samples")
        print(f"  (To process all, set TEST_MODE = False in script)")

    if not remaining_indices:
        print(f"\n✓ All samples already processed!")
        return

    # Process samples
    print(f"\n{'=' * 80}")
    print("PROCESSING")
    print('=' * 80)

    # Open output file (append if resuming, write new if starting fresh)
    mode = 'a' if processed else 'w'
    output_file_handle = open(OUTPUT_FILE, mode)

    # Track stats
    successful = 0
    failed = 0

    # Process each remaining sample
    for idx in tqdm(remaining_indices, desc="Relabeling"):
        sample = data[idx]

        # Extract text and entities from sample
        text = sample.get("text", "")
        entities = sample.get("entities", [])

        # Call LLM to relabel entities
        try:
            # This calls the Ollama API with the LLM prompt
            new_entities = call_ollama(text, entities)

            # Keep ALL original data from the sample, just add relabeled entities
            result = sample.copy()  # Start with complete original sample
            result["sample_idx"] = idx  # Add index for tracking
            result["original_entities"] = entities  # Keep original entities for comparison
            result["entities"] = new_entities  # Add relabeled entities

            # Write to NDJSON immediately (one line per sample)
            output_file_handle.write(json.dumps(result) + '\n')
            output_file_handle.flush()  # Force write to disk now

            successful += 1

        except Exception as e:
            # If LLM call fails, log error and keep original data
            tqdm.write(f"Sample {idx} failed: {str(e)[:60]}")

            # Keep original sample, mark as error
            result = sample.copy()
            result["sample_idx"] = idx
            result["original_entities"] = entities
            result["entities"] = []  # Empty entities on error
            result["error"] = str(e)

            # Write error to NDJSON too (so we can track failures)
            output_file_handle.write(json.dumps(result) + '\n')
            output_file_handle.flush()

            failed += 1

    # Close file when done
    output_file_handle.close()

    # Summary
    print(f"\n{'=' * 80}")
    print("✅ COMPLETE!")
    print('=' * 80)
    print(f"\nProcessing stats:")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {failed}")
    print(f"  Total processed: {successful + failed}")
    print(f"\nDataset stats:")
    print(f"  Input samples:  {len(data)}")
    print(f"  Output file:    {OUTPUT_FILE.name}")
    print(f"  Format:         NDJSON (one JSON object per line)")
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print('=' * 80)
    print(f"1. Review output: {OUTPUT_FILE}")
    print(f"2. Check sample comparisons")
    print(f"3. Convert to final format if needed")
    print("=" * 80)


if __name__ == "__main__":
    main()
