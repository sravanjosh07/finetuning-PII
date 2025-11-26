"""
Post-processing utilities for cleaning up model predictions.

Handles:
1. Entity deduplication within documents
2. Filtering code/technical context false positives
"""
import re


def deduplicate_entities(predictions):
    """
    Remove duplicate entities within a single document.
    Keeps the prediction with the highest confidence score.

    Args:
        predictions: List of prediction dictionaries with 'text', 'label', 'score' keys

    Returns:
        List of deduplicated predictions
    """
    if not predictions:
        return predictions

    # Group by normalized entity text (case-insensitive, stripped)
    entity_groups = {}

    for pred in predictions:
        # Normalize the entity text for comparison
        normalized_text = pred['text'].lower().strip()
        key = (normalized_text, pred['label'])  # Group by text + label combination

        # Keep the prediction with highest score
        if key not in entity_groups or pred['score'] > entity_groups[key]['score']:
            entity_groups[key] = pred

    # Return deduplicated predictions
    deduplicated = list(entity_groups.values())

    return deduplicated


def is_code_context(text):
    """
    Check if entity text looks like code/technical context.

    Patterns detected:
    - snake_case: contains underscores (field_name, user_id)
    - camelCase: mixed case without spaces (firstName, userId)
    - SCREAMING_SNAKE: all caps with underscores (MAX_VALUE)
    - Contains technical patterns like brackets, equals, dots in certain patterns

    Args:
        text: Entity text to check

    Returns:
        True if text appears to be code/technical context
    """
    # Check for underscore patterns (snake_case, SCREAMING_SNAKE)
    if '_' in text and len(text) > 2:
        return True

    # Check for camelCase (has both lower and upper, no spaces, starts with lower)
    if (any(c.islower() for c in text) and
        any(c.isupper() for c in text) and
        ' ' not in text and
        len(text) > 2 and
        text[0].islower()):
        return True

    # Check for patterns like "fieldName", "variableName" that end with common suffixes
    code_suffixes = ['Name', 'Type', 'Id', 'Number', 'Address', 'Date', 'Code', 'Key', 'Value']
    for suffix in code_suffixes:
        if text.endswith(suffix) and len(text) > len(suffix):
            # Check if there's a lowercase part before the suffix
            prefix = text[:-len(suffix)]
            if prefix and prefix[0].islower():
                return True

    return False


def filter_code_patterns(predictions):
    """
    Remove predictions that appear to be code/technical context.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Filtered list without code pattern matches
    """
    filtered = []

    for pred in predictions:
        if not is_code_context(pred['text']):
            filtered.append(pred)

    return filtered


def post_process_predictions(predictions, deduplicate=True, filter_code=True):
    """
    Apply all post-processing steps to predictions.

    Args:
        predictions: List of prediction dictionaries
        deduplicate: Whether to remove duplicate entities (default: True)
        filter_code: Whether to filter code patterns (default: True)

    Returns:
        Cleaned predictions list
    """
    result = predictions

    # Apply filtering first (before deduplication)
    if filter_code:
        result = filter_code_patterns(result)

    # Then deduplicate
    if deduplicate:
        result = deduplicate_entities(result)

    return result


# Statistics tracking
def get_post_processing_stats(original_predictions, processed_predictions):
    """
    Get statistics about what was removed during post-processing.

    Args:
        original_predictions: List of original predictions
        processed_predictions: List after post-processing

    Returns:
        Dictionary with statistics
    """
    original_count = len(original_predictions)
    processed_count = len(processed_predictions)
    removed_count = original_count - processed_count

    # Track what was removed
    processed_texts = {p['text'].lower().strip() for p in processed_predictions}
    removed = [p for p in original_predictions
               if p['text'].lower().strip() not in processed_texts]

    # Categorize removals
    duplicates = original_count - len({(p['text'].lower().strip(), p['label'])
                                       for p in original_predictions})
    code_patterns = sum(1 for p in original_predictions if is_code_context(p['text']))

    return {
        'original_count': original_count,
        'processed_count': processed_count,
        'removed_count': removed_count,
        'removed_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
        'duplicates_removed': duplicates,
        'code_patterns_removed': code_patterns,
    }
