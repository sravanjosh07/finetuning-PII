"""
Centralized model configuration
Single source of truth for all GLiNER PII models
"""
from pathlib import Path

# Get the project root directory (parent of utils)
UTILS_DIR = Path(__file__).parent
PROJECT_ROOT = UTILS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent  # Go up to fintuning_PII directory

# Urchade models
URCHADE_LABELS = [
    "person", "organization", "phone number", "address", "email",
    "passport number", "credit card number", "social security number",
    "date of birth", "blood type", "driver's license", "identity card",
    "birth certificate", "bank account number", "CVV", "IBAN",
    "mobile phone number", "fax number", "health insurance ID",
    "medical condition", "medication", "tax ID", "CPF", "CNPJ",
    "IP address", "username", "digital signature", "social media handle",
    "flight number", "train ticket", "license plate", "vehicle registration",
    "reservation number", "transaction number"
]

# Gretel models
GRETEL_LABELS = [
    "medical_record_number", "date_of_birth", "ssn", "date", "first_name",
    "email", "last_name", "customer_id", "employee_id", "name", "street_address",
    "phone_number", "ipv4", "credit_card_number", "license_plate", "address",
    "user_name", "device_identifier", "bank_routing_number", "date_time",
    "company_name", "unique_identifier", "biometric_identifier", "account_number",
    "city", "certificate_license_number", "time", "postcode", "vehicle_identifier",
    "coordinate", "country", "api_key", "ipv6", "password",
    "health_plan_beneficiary_number", "national_id", "tax_id", "url", "state",
    "swift_bic", "cvv", "pin"
]

# NVIDIA models
NVIDIA_LABELS = [
    "first_name", "last_name", "date_of_birth", "age", "gender",
    "phone_number", "email", "fax_number", "ssn", "bank_routing_number",
    "swift_bic", "credit_debit_card", "account_number", "cvv",
    "health_plan_beneficiary_number", "street_address", "city", "state",
    "county", "postcode", "country", "coordinate", "vehicle_identifier",
    "license_plate", "device_identifier", "employee_id", "customer_id",
    "medical_record_number", "biometric_identifier", "certificate_license_number",
    "mac_address", "ipv4", "occupation", "education_level", "employment_status",
    "blood_type", "race_ethnicity", "religious_belief", "political_view",
    "sexuality", "user_name", "password", "pin", "http_cookie",
    "date", "time", "date_time", "company_name", "url", "language"
]

# Knowledgator models
KNOWLEDGATOR_LABELS = [
    "name", "first name", "last name", "name (medical professional)",
    "date of birth", "age", "gender", "marital status",
    "email address", "phone number", "IP address", "URL",
    "location address", "location street", "location city",
    "location state", "location country", "location zip",
    "account number", "bank account", "routing number",
    "credit card", "credit card expiration", "CVV", "SSN", "money",
    "condition", "medical process", "drug", "dose", "blood type", "injury",
    "organization (medical facility)", "healthcare number", "medical code",
    "passport number", "driver license", "username", "password", "vehicle ID"
]

# Finetuned model (custom trained from cam_pii)
# Labels are built from subcategories: entity.name.lower().replace("_", " ")
FINETUNED_LABELS = [
    # Personal Information
    "date of birth", "social security number", "tax identification number",
    "drivers license number", "identity card number", "passport number", "full name",
    # Contact Information
    "phone number", "mobile phone number", "address", "email address",
    "ip address", "fax number",
    # Financial Information
    "credit card number", "credit score", "bank account number",
    "bank account balance", "iban",
    # Medical Information
    "health insurance id number", "insurance plan number",
    "national health insurance number", "medical condition",
    "medication", "birth certificate number", "medical treatment",
    # Identification Information
    "student id number", "username",
    # Miscellaneous Information #TODO: we would not need these explicitely!
    "year", "city", "location", "person", "description", "body organ",
    "food", "object", "chemical process", "abstract concept",
    "biological entity", "art work", "utility work", "action",
    "societal entity", "organization"
]

# Updated label set - cleaned up for better accuracy
# Changes from FINETUNED_LABELS:
# - "date of birth" → "date" (covers all date types: birth, appointment, transaction, etc.)
# - Removed non-PII labels: "year", "city", "location", "person", "description"
# - Removed unused labels: "body organ", "food", "object", "chemical process", "abstract concept",
#   "biological entity", "art work", "utility work", "action", "societal entity"
# - Kept "organization" (133 entities in test set, 9.62%)
UPDATED_FINETUNED_LABELS = [
    # Personal Information
    "date",  # Changed from "date of birth" - covers all date types
    "social security number", "tax identification number",
    "drivers license number", "identity card number", "passport number", "full name",
    # Contact Information
    "phone number", "mobile phone number", "address", "email address",
    "ip address", "fax number",
    # Financial Information
    "credit card number", "credit score", "bank account number",
    "bank account balance", "iban",
    # Medical Information
    "health insurance id number", "insurance plan number",
    "national health insurance number", "medical condition",
    "medication", "birth certificate number", "medical treatment",
    # Identification Information
    "student id number", "username",
    # Organization Information
    "organization",
]


"""here is my transaction ID: 1234567890"""
# Simplified 21-label set - consolidates similar types for better GLiNER performance
# Based on: GLiNER paper (NAACL 2024), Azure PII, Presidio naming conventions
# Key changes:
# - Merged ID types → "government id" (passport, driver's license, national ID, student ID)
# - Merged health insurance types → "health insurance number"
# - Simplified medical labels (no "/" - cleaner for semantic embeddings)
# - Labels in lower case (GLiNER recommendation)
SIMPLIFIED_24_LABELS = [
    # Personal
    "date",
    "full name",
    "username",

    # Government/Official IDs
    "social security number",
    "tax identification number",
    "passport number",
    "driver's license number",
    "identification number",  # catch-all: national ID, student ID, birth certificate, etc.

    # Contact
    "phone number",
    "address",
    "email address",
    "ip address",
    "fax number",

    # Financial
    "credit card number",
    "credit score",
    "bank account number",
    "amount",
    "iban",
    "health insurance number",  # all types: health, auto, home, life, travel, etc.

    # Medical
    "medical condition",  # diseases, injuries, disorders, syndromes
    "medication",
    "medical treatment",

    # Organization
    "organization",
]

# Mapping from various label formats to SIMPLIFIED_24_LABELS
# This is the single source of truth for label consolidation
LABEL_CONSOLIDATION_MAP = {
    # Driver's license - keep separate
    "drivers license number": "driver's license number",
    "driver_license": "driver's license number",
    "drivers_license_number": "driver's license number",

    # Passport - keep separate
    "passport": "passport number",
    "passport_number": "passport number",

    # Other IDs - catch-all
    "identity card number": "identification number",
    "identity_card_number": "identification number",
    "birth certificate number": "identification number",
    "birth_certificate_number": "identification number",
    "student id number": "identification number",
    "student_id_number": "identification number",
    "national id": "identification number",
    "national_id": "identification number",

    # Insurance - all types to general "insurance number"
    "health insurance id number": "insurance number",
    "health_insurance_id_number": "insurance number",
    "health_insurance": "insurance number",
    "health insurance number": "insurance number",
    "insurance plan number": "insurance number",
    "insurance_plan_number": "insurance number",
    "national health insurance number": "insurance number",
    "national_health_insurance_number": "insurance number",

    # Phone numbers
    "mobile phone number": "phone number",
    "mobile_phone_number": "phone number",
    "fax_number": "fax number",

    # Financial
    "ssn": "social security number",
    "social_security_number": "social security number",
    "tax_id": "tax identification number",
    "tax_identification_number": "tax identification number",
    "credit_card": "credit card number",
    "credit_card_number": "credit card number",
    "credit_score": "credit score",
    "bank_account": "bank account number",
    "bank_account_number": "bank account number",
    "bank account balance": "amount",
    "bank_account_balance": "amount",

    # Contact
    "email": "email address",
    "email_address": "email address",
    "ip_address": "ip address",

    # Medical
    "medical_condition": "medical condition",
    "disease": "medical condition",
    "procedure": "medical treatment",
    "medical_treatment": "medical treatment",
}


# Model registry - single source of truth
MODELS = {
    # === Active Models ===
    'Finetuned-PII': {
        'path': str(REPO_ROOT / 'finetuned_gliner'),
        'labels': UPDATED_FINETUNED_LABELS,  # Using 28 updated labels
        'group': 'finetuned',
        'description': 'Custom finetuned PII model (28 labels)'
    },
    'Gretel-Large': {
        'path': 'gretelai/gretel-gliner-bi-large-v1.0',
        'labels': GRETEL_LABELS,
        'group': 'gretel',
        'description': 'Gretel large model - highest accuracy (42 labels)'
    },
    'Knowledge-Large': {
        'path': 'knowledgator/gliner-pii-large-v1.0',
        'labels': KNOWLEDGATOR_LABELS,
        'group': 'knowledgator',
        'description': 'Knowledgator large model - highest capacity (40 labels)'
    },
    'E3-JSI-Domains': {
        'path': 'E3-JSI/gliner-multi-pii-domains-v1',
        'labels': URCHADE_LABELS,  # Based on urchade, uses similar labels
        'group': 'e3-jsi',
        'description': 'E3-JSI multi-domain PII model (9 languages, 40+ PII types)'
    },
    'Urchade-PII': {
        'path': 'urchade/gliner_multi_pii-v1',
        'labels': URCHADE_LABELS,
        'group': 'urchade',
        'description': 'Urchade multilingual PII model (34 labels)'
    },
    'Gretel-Base': {
        'path': 'gretelai/gretel-gliner-bi-base-v1.0',
        'labels': GRETEL_LABELS,
        'group': 'gretel',
        'description': 'Gretel base model (42 labels)'
    },
    'Gretel-Small': {
        'path': 'gretelai/gretel-gliner-bi-small-v1.0',
        'labels': GRETEL_LABELS,
        'group': 'gretel',
        'description': 'Gretel small model (42 labels)'
    },
    'Knowledge-Base': {
        'path': 'knowledgator/gliner-pii-base-v1.0',
        'labels': KNOWLEDGATOR_LABELS,
        'group': 'knowledgator',
        'description': 'Knowledgator base model (40 labels)'
    },
    'Knowledge-Small': {
        'path': 'knowledgator/gliner-pii-small-v1.0',
        'labels': KNOWLEDGATOR_LABELS,
        'group': 'knowledgator',
        'description': 'Knowledgator small model (40 labels)'
    },

    # === Inactive Models (commented out) ===
    # 'NVIDIA-PII': {
    #     'path': 'nvidia/gliner-PII',
    #     'labels': NVIDIA_LABELS,
    #     'group': 'nvidia',
    #     'description': 'NVIDIA Nemotron PII model (55 labels)'
    # },
}


def get_models_by_group(group_name):
    """Get all models in a specific group"""
    return {name: config for name, config in MODELS.items()
            if config['group'] == group_name}


def get_model_names():
    """Get list of all model names"""
    return list(MODELS.keys())


def get_groups():
    """Get list of all groups"""
    return list(set(config['group'] for config in MODELS.values()))
