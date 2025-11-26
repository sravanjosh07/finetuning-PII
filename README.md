# Finetuned PII Detection v2

## Structure

```
finetuned-pii-2/
├── tests/                           # Evaluation scripts
│   ├── compare_all_models.py        # Each model uses native labels
│   ├── compare_all_models_unified.py # All models use same labels
│   └── compare_models_unified_labels.py  # Old unified comparison
├── utils/                           # Shared utilities (clean!)
│   ├── models_config.py            # Model configurations
│   ├── evaluate.py                 # Evaluation functions
│   ├── label_normalizer.py         # Label normalization
│   ├── post_processing.py          # Deduplication & code filtering
│   └── relabel_with_llm.py         # LLM-based ground truth relabeling
├── data_creation/                   # Dataset creation scripts
│   ├── extract_medical_samples.py  # Extract labeled medical samples
│   ├── extract_unlabeled_medical_text.py # Find unlabeled samples
│   ├── label_with_model.py         # Label with finetuned model
│   └── scan_medical_datasets.py    # Scan for unique labels
├── notebooks/                       # Jupyter notebooks
│   ├── explore_gold_dataset.ipynb  # Analyze gold dataset
│   ├── simple_model_test.ipynb     # Quick 50-sample test
│   └── test_balanced_dataset.ipynb # Test on balanced data
├── data/                            # Test datasets and samples
├── results/                         # Results from native labels
├── results_unified/                 # Results from unified labels
├── preds/                           # Predictions from native labels
└── preds_unified/                   # Predictions from unified labels
```

## Resources

- **Models**: `../finetuned_gliner/` - Trained PII models
- **Test Data**: `../Data/gold_testdataset_27labels.ndjson` - Gold test dataset
- **Full Datasets**: `../Data/` - Complete dataset collection
- **Old Code**: `../old_code_archive/` - Previous work and scripts

## Evaluation Scripts

### compare_all_models.py
Tests each model with its **native labels**. Shows how well each model performs with the labels it was trained on.

**Usage:**
```bash
python tests/compare_all_models.py
```

**Output:**
- `results/` - Metrics and comparisons
- `preds/` - Full predictions for each model

### compare_all_models_unified.py
Tests all models with the **same labels** (UPDATED_FINETUNED_LABELS). Fair comparison of model architectures independent of label differences.

**Usage:**
```bash
python tests/compare_all_models_unified.py
```

**Output:**
- `results_unified/` - Metrics and comparisons
- `preds_unified/` - Full predictions for each model

## Notebooks

### explore_gold_dataset.ipynb
Analyze the gold test dataset - label counts, distribution, examples.

### simple_model_test.ipynb
Quick test with 50 samples from gold dataset.

### test_balanced_dataset.ipynb
Test on balanced dataset (100 samples per class).

### confusion_matrix_from_predictions.ipynb
Load saved predictions and create confusion matrices. Shows:
- 21 consolidated labels confusion matrix
- Most confused label pairs
- Per-label accuracy
- Examples of over-detection (false positives)
- Examples of under-detection (missed entities)
- Identifies which entities the model struggles with
- Configurable threshold filtering to reduce false positives

## Post-Processing

### Deduplication & Code Filtering
Reduce false positives by:
1. **Entity deduplication** - Remove duplicate entities within documents (keeps highest confidence)
2. **Code pattern filtering** - Remove code/technical context (snake_case, camelCase, field names)

Enable in evaluation scripts by editing configuration:
```python
# In compare_all_models.py or compare_all_models_unified.py
ENABLE_POST_PROCESSING = True  # Set to True to enable
DEDUPLICATE = True              # Remove duplicate entities
FILTER_CODE = True              # Filter code patterns
```

This reduces false positives like:
- Repeated name mentions (Omar Aponte detected 10+ times → deduplicated to 1)
- Field names in code (credit_card_number, user_id → removed)
- Variable names (firstName, lastName → removed)

## Dataset Creation

### extract_medical_samples.py
**Main script** - Clean, simple extraction of medical samples:
- Scans medical datasets for unique labels
- Maps dataset labels to our 27 standard PII labels intelligently
- Extracts 200 samples with medical treatment entities
- Saves label mapping for future use (training datasets)

**Usage:**
```bash
python data_creation/extract_medical_samples.py
```

**Output:**
- `data/medical_treatment_samples.ndjson` - 46 labeled medical samples
- `data/label_mapping.json` - Smart label mapping for future use

### extract_unlabeled_medical_text.py
Find unlabeled clinical text with treatment keywords:
- Searches agbonnet (30K) and asclepius (158K) datasets
- Finds samples with surgery, procedures, therapy keywords
- Found 2,125+ candidates, extracts 100 by default

**Usage:**
```bash
python data_creation/extract_unlabeled_medical_text.py
```

**Output:**
- `data/unlabeled_medical_treatment_text.ndjson` - 100 unlabeled samples

### label_with_model.py
Label unlabeled samples using your finetuned model:
- Takes unlabeled samples and runs model predictions
- Combines with existing labeled samples
- Reaches 200+ medical treatment samples

**Usage:**
```bash
python data_creation/label_with_model.py
```

**Output:**
- `data/medical_treatment_samples_combined.ndjson` - 200+ samples ready to use

### scan_medical_datasets.py
Utility to explore datasets:
- Scans all medical datasets for unique labels
- Shows label distribution and counts
- Helps understand what data is available

**Usage:**
```bash
python data_creation/scan_medical_datasets.py
```

## Ground Truth Relabeling

### relabel_with_llm.py
Fix annotation issues in the gold test dataset using LLM (qwen2.5:32b):
- Fill in missing PII entities (samples with 0 entities that should have some)
- Normalize labels to 27 standard labels
- Merge adjacent name entities (first_name + last_name → full name)
- Remove invalid labels (job titles, ticket numbers)

**Usage:**
```bash
# Full run (recommended for weekend - ~6-8 hours for 1772 samples)
python utils/relabel_with_llm.py

# Test mode (5 samples)
TEST_MODE=true python utils/relabel_with_llm.py

# Use faster 14b model instead of 32b
OLLAMA_MODEL=qwen2.5:14b python utils/relabel_with_llm.py

# Resume after interruption (automatically detects progress)
python utils/relabel_with_llm.py
```

**Output:**
- `data/gold_testdataset_llm_relabeled.ndjson` - Cleaned and completed annotations
- Supports resume if interrupted
- Shows progress with ETA

**Requirements:**
- Ollama running locally: `ollama serve`
- Model pulled: `ollama pull qwen2.5:32b`
- 48GB RAM recommended for 32b model
