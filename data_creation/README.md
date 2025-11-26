# Dataset Creation Scripts

Simple scripts for analyzing and augmenting the PII test dataset.

## Our 26 Labels

After merging mobile phone + phone number:

```python
[
    "date", "full name",
    "social security number", "tax identification number",
    "drivers license number", "identity card number", "passport number",
    "birth certificate number", "student id number",
    "phone number", "fax number",  # mobile merged into phone
    "email address", "ip address", "address",
    "credit card number", "credit score", "bank account number",
    "amount",  # changed from "bank account balance"
    "iban",
    "health insurance id number", "insurance plan number",
    "national health insurance number",
    "medical condition", "medication", "medical treatment",
    "username", "organization"
]
```

## Quick Workflow

### 1. Extract Minority Label Samples (Optional - already done!)

```bash
python data_creation/extract_minority_labels.py
```

**What it does:**
- Extracts samples for underrepresented labels from e3jsi and combined medical datasets
- Creates separate JSON file for each label (modular approach)

**Already extracted:**
- medical treatment: 100 samples, 150 entities
- medication: 50 samples, 149 entities
- amount: 62 samples, 62 entities

**Output:** `data/minority_labels/` with individual files

### 2. Combine into Final Dataset

```bash
python data_creation/combine_with_balanced_dataset.py
```

**What it does:**
- Loads balanced_test_100_per_class_27_labels_filtered_300tok.json (1771 samples)
- Adds minority label samples (medical treatment, medication, amount)
- Converts to matching JSON format with all keys (entity, types, start, end, original_type, canonical_type, source_dataset)
- Removes duplicates by text
- Saves final augmented dataset

**Configuration:** Edit `MINORITY_LABELS_TO_ADD` in script to choose which labels to include

**Output:** `data/balanced_augmented_test_dataset.json`

**Final stats:**
- Balanced dataset: 1771 samples
- Added: 212 samples (medical treatment + medication + amount)
- Duplicates removed: 156 samples
- **Final: 1827 samples**

## Directory Structure

```
data/
├── minority_labels/                          # Modular minority label files
│   ├── medical_treatment_samples.json        # 100 samples, 150 entities
│   ├── medication_samples.json               # 50 samples, 149 entities
│   ├── amount_samples.json                   # 62 samples, 62 entities
│   ├── mobile_phone_number_samples.json      # 4 samples
│   ├── insurance_plan_number_samples.json    # 2 samples
│   └── README.json                           # Index with stats
├── unlabeled/                                # Unlabeled text for future use
│   └── unlabeled_medical_treatment_text.ndjson  # 300 samples
├── label_mapping.json                        # Smart label mapping reference
└── balanced_augmented_test_dataset.json      # Final dataset (1827 samples) ⭐
```

**Note:**
- Uses balanced_test_100_per_class_27_labels_filtered_300tok.json as base
- Gold dataset will be generated separately via LLM relabeling (utils/relabel_with_llm.py)

## Label Changes

### 1. Merged: mobile phone number + phone number → "phone number"
**Reason:** GLiNER cannot distinguish mobile vs landline from context (only 8.4% have "mobile"/"cell" explicitly mentioned)
**Result:** 417 total phone numbers in gold dataset (334 + 83)

### 2. Changed: "bank account balance" → "amount"
**Reason:** More flexible, covers all financial amounts (not just account balances)
**Result:** 58 samples in gold dataset

## Label Mapping

The `extract_minority_labels.py` script includes smart label mapping:

**Medical:**
- "treatment", "procedure", "therapy", "surgery" → "medical treatment"
- "drug", "prescription" → "medication"

**Financial:**
- "balance", "bank account balance", "salary" → "amount"

**Multilingual support:**
- French: "nom", "adresse", "bedrag"
- Dutch: "persoon", "medische procedure"
- Italian, Greek, etc.

## Notes

- **Medical treatment & medication:** Already extracted from combined dataset (e3jsi labeled + model labeled from asclepius)
- **Credit score:** No data available in datasets (searched Gretel finance, found none)
- **Birth certificate, student ID:** Rare by nature, low priority
- **Phone numbers:** Now merged (mobile + phone = 417 total)
