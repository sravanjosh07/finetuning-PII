"""
Test LLM Relabeling on Balanced Dataset

Simple test script - no arguments, just edit NUM_SAMPLES below to test more/less

Usage:
    python utils/test_relabel_llm.py
"""
import json
import sys
from pathlib import Path

# Add project to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import the relabel functions
from utils.relabel_with_llm import call_ollama, LABELS, MODEL, OLLAMA_URL
import requests

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# How many samples to test (default: 5)
NUM_SAMPLES = 5

# Input and output files
INPUT_FILE = PROJECT_DIR / "data" / "balanced_augmented_test_dataset.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "test_relabel_output.ndjson"


def main():
    print("=" * 80)
    print("TEST LLM RELABELING - BALANCED DATASET")
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

    # Take first NUM_SAMPLES
    test_samples = data[:NUM_SAMPLES]
    print(f"\n⚠ TEST MODE: Processing {len(test_samples)} samples")

    # Process samples (write NDJSON as we go)
    print(f"\n{'=' * 80}")
    print("PROCESSING")
    print('=' * 80)

    # Open output file for writing (append mode for resume support)
    output_file_handle = open(OUTPUT_FILE, 'w')

    results = []
    successful = 0
    failed = 0

    for idx, sample in enumerate(test_samples):
        print(f"\n[{idx+1}/{len(test_samples)}] Processing...")

        text = sample.get("text", "")
        entities = sample.get("entities", [])

        # Show input
        print(f"  Text preview: {text[:80]}...")
        print(f"  Input entities: {len(entities)}")

        # Call LLM to relabel
        try:
            new_entities = call_ollama(text, entities)
            print(f"  Output entities: {len(new_entities)}")

            # Show first few entities
            if new_entities:
                print(f"  Sample outputs:")
                for e in new_entities[:3]:
                    print(f"    - {e.get('text', ''):<30} → {e.get('label', '')}")

            # Keep ALL original data, just add relabeled entities
            result = sample.copy()  # Complete original sample
            result["sample_idx"] = idx  # Add index for tracking
            result["original_entities"] = entities  # Original entities for comparison
            result["entities"] = new_entities  # Relabeled entities (overwrites original)

            # Write to NDJSON immediately
            output_file_handle.write(json.dumps(result) + '\n')
            output_file_handle.flush()  # Ensure written to disk

            results.append(result)
            successful += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")

            # Keep original sample, mark as error
            result = sample.copy()
            result["sample_idx"] = idx
            result["original_entities"] = entities
            result["entities"] = []  # Empty on error
            result["error"] = str(e)

            # Write error to NDJSON too
            output_file_handle.write(json.dumps(result) + '\n')
            output_file_handle.flush()

            results.append(result)
            failed += 1

    output_file_handle.close()

    print(f"\n{'=' * 80}")
    print("SAVED RESULTS")
    print('=' * 80)
    print(f"✓ Saved to: {OUTPUT_FILE.name} (NDJSON format)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)

    print(f"Total samples:  {len(results)}")
    print(f"Successful:     {successful}")
    print(f"Failed:         {failed}")

    # Show comparison
    print(f"\n{'=' * 80}")
    print("SAMPLE COMPARISON (First sample)")
    print('=' * 80)

    if results and "error" not in results[0]:
        result = results[0]
        print(f"\nText: {result['text'][:100]}...")
        print(f"Source: {result.get('source_dataset', 'unknown')}")

        print(f"\nOriginal entities ({len(result['original_entities'])}):")
        for e in result['original_entities'][:5]:
            entity_text = e.get('entity', '')
            entity_types = e.get('types', [])
            label = entity_types[0] if entity_types else 'unknown'
            print(f"  - {entity_text:<30} → {label}")

        print(f"\nRelabeled entities ({len(result['entities'])}):")
        for e in result['entities'][:5]:
            print(f"  - {e.get('text', ''):<30} → {e.get('label', '')}")

        print(f"\nDifference: {len(result['entities']) - len(result['original_entities'])} entities (negative = merged names)")

    print(f"\n{'=' * 80}")
    print("✅ TEST COMPLETE!")
    print('=' * 80)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Review the results before running on full dataset.")
    print("=" * 80)


if __name__ == "__main__":
    main()
