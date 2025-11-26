"""
Run two scripts in sequence

This script runs relabel_balanced_dataset.py first, then verify_negative_examples.py
Perfect for weekend runs when you won't be around to start the second script.
"""
import subprocess
import sys
from datetime import datetime

print("=" * 80)
print("SEQUENTIAL SCRIPT RUNNER")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Script 1: Relabel balanced dataset
print("=" * 80)
print("STEP 1: Running relabel_balanced_dataset.py")
print("=" * 80)
step1_start = datetime.now()

try:
    result = subprocess.run(
        [sys.executable, "utils/relabel_balanced_dataset.py"],
        check=True
    )
    step1_end = datetime.now()
    step1_duration = step1_end - step1_start

    print("\n" + "=" * 80)
    print(f"✓ STEP 1 COMPLETE")
    print(f"  Duration: {step1_duration}")
    print("=" * 80)
    print()

except subprocess.CalledProcessError as e:
    print("\n" + "=" * 80)
    print(f"✗ STEP 1 FAILED")
    print(f"  Error code: {e.returncode}")
    print("=" * 80)
    print("\nStopping - not running step 2")
    sys.exit(1)

except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    sys.exit(1)


# Script 2: Verify negative examples
print("=" * 80)
print("STEP 2: Running verify_negative_examples.py")
print("=" * 80)
step2_start = datetime.now()

try:
    result = subprocess.run(
        [sys.executable, "utils/verify_negative_examples.py"],
        check=True
    )
    step2_end = datetime.now()
    step2_duration = step2_end - step2_start

    print("\n" + "=" * 80)
    print(f"✓ STEP 2 COMPLETE")
    print(f"  Duration: {step2_duration}")
    print("=" * 80)

except subprocess.CalledProcessError as e:
    print("\n" + "=" * 80)
    print(f"✗ STEP 2 FAILED")
    print(f"  Error code: {e.returncode}")
    print("=" * 80)
    sys.exit(1)

except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    sys.exit(1)


# Final summary
total_end = datetime.now()
total_duration = total_end - step1_start

print("\n" + "=" * 80)
print("ALL STEPS COMPLETE!")
print("=" * 80)
print(f"Step 1 duration: {step1_duration}")
print(f"Step 2 duration: {step2_duration}")
print(f"Total duration:  {total_duration}")
print(f"Finished at:     {total_end.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
