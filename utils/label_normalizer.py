"""
Label Normalizer: Clean, centralized label normalization system

Maps all model labels and test data labels to a common canonical format.
The canonical format is based on the 23 test data labels for consistency.

Updated: 2024-11 - Aligned with gold_test_24labels test dataset
"""

from typing import Dict, List, Set, Tuple, Optional
import json


class LabelNormalizer:
    """
    Single source of truth for all label normalization.

    The 23 TEST DATA labels are our canonical reference:
    - address, amount, bank account number, credit card number, credit score
    - date, driver's license number, email address, fax number, full name
    - iban, identification number, insurance number, ip address
    - medical condition, medical treatment, medication
    - organization, passport number, phone number
    - social security number, tax identification number, username

    All model labels map TO these canonical labels.
    """

    # ========================================================================
    # CANONICAL LABELS - Based on TEST DATA (23 labels)
    # ========================================================================
    # These are the exact labels used in gold_test_24labels_2300sampels.ndjson
    # All model predictions and ground truth normalize to these labels

    CANONICAL_LABELS = {
        # === NAMES ===
        "full name": [
            "full name", "name", "person", "person name", "full_name", "fullname",
            "first_name", "last_name", "first name", "last name", "firstname", "lastname",
            "name (medical professional)", "given_name", "surname", "family_name"
        ],

        # === DATES ===
        "date": [
            "date", "date_of_birth", "date of birth", "dob", "birthday", "birthdate",
            "birth_date", "birth date", "date_time", "datetime", "time", "timestamp",
            "expiration date", "start date", "end date", "event date"
        ],

        # === GOVERNMENT IDs ===
        "social security number": [
            "social security number", "ssn", "social_security_number", "social security"
        ],
        "tax identification number": [
            "tax identification number", "tax_id", "tax id", "tin",
            "tax_identification_number"
        ],
        "driver's license number": [
            "driver's license number", "drivers license number", "driver_license",
            "driver's license", "driver's_license", "drivers_license", "driving_license",
            "license_number", "dl_number", "dl number", "driver license"
        ],
        "passport number": [
            "passport number", "passport", "passport_number"
        ],
        "identification number": [
            "identification number", "identity card number", "identity_card_number",
            "identity card", "identity_card", "national_id", "national id",
            "id number", "id card", "government id", "birth certificate number",
            "birth_certificate_number", "birth certificate", "student id number",
            "student_id", "certificate_license_number", "unique_identifier"
        ],

        # === CONTACT ===
        "phone number": [
            "phone number", "phone_number", "phone", "mobile phone number",
            "mobile_phone_number", "mobile phone", "mobile", "telephone",
            "contact number", "cell phone", "cell"
        ],
        "fax number": [
            "fax number", "fax_number", "fax"
        ],
        "email address": [
            "email address", "email_address", "email", "e-mail"
        ],
        "address": [
            "address", "street_address", "street address", "location address",
            "location", "mailing address", "home address", "full_address",
            "location street", "location city", "location state", "location country",
            "location zip", "city", "state", "country", "postcode", "zipcode",
            "coordinate", "coordinates"
        ],
        "ip address": [
            "ip address", "ip_address", "ip", "ipv4", "ipv6", "IP address"
        ],

        # === FINANCIAL ===
        "credit card number": [
            "credit card number", "credit_card_number", "credit card", "credit_card",
            "creditcard", "creditcardnumber", "card_number", "credit_debit_card"
        ],
        "credit score": [
            "credit score", "credit_score", "credit rating"
        ],
        "bank account number": [
            "bank account number", "bank_account_number", "bank account", "bank_account",
            "account_number", "account number", "account", "checking account",
            "savings account", "bank_routing_number", "routing_number", "routing number"
        ],
        "iban": [
            "iban", "IBAN", "international_bank_account_number", "swift_bic", "swift", "bic"
        ],
        "amount": [
            "amount", "money", "monetary amount", "currency", "balance",
            "bank account balance", "bank_account_balance", "loan amount",
            "transaction amount", "salary", "account balance"
        ],

        # === INSURANCE ===
        "insurance number": [
            "insurance number", "health insurance id number", "health_insurance_id_number",
            "health insurance ID", "health insurance", "insurance_number",
            "health_plan_beneficiary_number", "healthcare number", "healthcare_number",
            "national health insurance number", "insurance plan number", "insurance id"
        ],

        # === MEDICAL ===
        "medical condition": [
            "medical condition", "medical_condition", "condition", "diagnosis",
            "disease", "health condition", "injury", "symptom"
        ],
        "medication": [
            "medication", "drug", "medicine", "prescription", "dose"
        ],
        "medical treatment": [
            "medical treatment", "medical_treatment", "treatment", "procedure",
            "medical procedure", "medical_procedure", "medical process", "therapy"
        ],

        # === ORGANIZATION ===
        "organization": [
            "organization", "company_name", "company name", "company", "org",
            "organization name", "business_name", "corporation", "companyname",
            "organization (medical facility)", "institution", "bank", "bank name"
        ],

        # === ONLINE/TECH ===
        "username": [
            "username", "user_name", "user name", "userid", "user_id", "login"
        ],
    }

    # ========================================================================
    # MODEL-SPECIFIC LABEL MAPPINGS
    # ========================================================================
    # Maps model's native labels → canonical labels (test data format)
    # Only needed for labels that don't auto-match via CANONICAL_LABELS

    MODEL_MAPPINGS = {
        # Finetuned model - most labels already match test data format
        "Finetuned-PII": {
            "drivers license number": "driver's license number",
            "identity card number": "identification number",
            "birth certificate number": "identification number",
            "student id number": "identification number",
            "mobile phone number": "phone number",
            "health insurance id number": "insurance number",
            "insurance plan number": "insurance number",
            "national health insurance number": "insurance number",
            "bank account balance": "amount",
        },

        # E3-JSI / Urchade - uses descriptive label names
        "E3-JSI-Domains": {
            "person": "full name",
            "email": "email address",
            "date of birth": "date",
            "driver's license": "driver's license number",
            "identity card": "identification number",
            "birth certificate": "identification number",
            "CVV": "credit card number",  # Group with credit card
            "IBAN": "iban",
            "IP address": "ip address",
            "tax ID": "tax identification number",
            "health insurance ID": "insurance number",
            "CPF": "identification number",
            "CNPJ": "identification number",
            "blood type": "medical condition",
            "digital signature": "identification number",
            "social media handle": "username",
            "flight number": "identification number",
            "train ticket": "identification number",
            "license plate": "identification number",
            "vehicle registration": "identification number",
            "reservation number": "identification number",
            "transaction number": "identification number",
        },

        # Gretel - uses snake_case labels
        "Gretel-Base": {
            "first_name": "full name",
            "last_name": "full name",
            "name": "full name",
            "date_of_birth": "date",
            "date_time": "date",
            "time": "date",
            "ssn": "social security number",
            "tax_id": "tax identification number",
            "national_id": "identification number",
            "certificate_license_number": "identification number",
            "unique_identifier": "identification number",
            "phone_number": "phone number",
            "email": "email address",
            "street_address": "address",
            "city": "address",
            "state": "address",
            "country": "address",
            "postcode": "address",
            "coordinate": "address",
            "ipv4": "ip address",
            "ipv6": "ip address",
            "credit_card_number": "credit card number",
            "cvv": "credit card number",
            "account_number": "bank account number",
            "bank_routing_number": "bank account number",
            "swift_bic": "iban",
            "health_plan_beneficiary_number": "insurance number",
            "medical_record_number": "insurance number",
            "company_name": "organization",
            "user_name": "username",
            "password": "username",  # Group credentials
            "pin": "username",
            "api_key": "username",
            "device_identifier": "identification number",
            "vehicle_identifier": "identification number",
            "biometric_identifier": "identification number",
            "employee_id": "identification number",
            "customer_id": "identification number",
            "license_plate": "identification number",
            "url": "ip address",  # Web identifier
        },
        "Gretel-Small": {},  # Same as Gretel-Base
        "Gretel-Large": {},  # Same as Gretel-Base

        # Knowledgator - uses space-separated labels
        "Knowledge-Base": {
            "name": "full name",
            "first name": "full name",
            "last name": "full name",
            "name (medical professional)": "full name",
            "date of birth": "date",
            "age": "date",  # Age relates to date
            "gender": "full name",  # Demographic
            "marital status": "full name",  # Demographic
            "email address": "email address",
            "phone number": "phone number",
            "IP address": "ip address",
            "URL": "ip address",
            "location address": "address",
            "location street": "address",
            "location city": "address",
            "location state": "address",
            "location country": "address",
            "location zip": "address",
            "account number": "bank account number",
            "bank account": "bank account number",
            "routing number": "bank account number",
            "credit card": "credit card number",
            "credit card expiration": "credit card number",
            "CVV": "credit card number",
            "SSN": "social security number",
            "money": "amount",
            "condition": "medical condition",
            "injury": "medical condition",
            "blood type": "medical condition",
            "drug": "medication",
            "dose": "medication",
            "medical process": "medical treatment",
            "organization (medical facility)": "organization",
            "healthcare number": "insurance number",
            "medical code": "insurance number",
            "passport number": "passport number",
            "driver license": "driver's license number",
            "username": "username",
            "password": "username",
            "vehicle ID": "identification number",
        },
        "Knowledge-Small": {},  # Same as Knowledge-Base
        "Knowledge-Large": {},  # Same as Knowledge-Base

        # Urchade (same as E3-JSI)
        "Urchade-PII": {},  # Same as E3-JSI-Domains
    }

    # Copy mappings for model variants
    MODEL_MAPPINGS["Gretel-Small"] = MODEL_MAPPINGS["Gretel-Base"].copy()
    MODEL_MAPPINGS["Gretel-Large"] = MODEL_MAPPINGS["Gretel-Base"].copy()
    MODEL_MAPPINGS["Knowledge-Small"] = MODEL_MAPPINGS["Knowledge-Base"].copy()
    MODEL_MAPPINGS["Knowledge-Large"] = MODEL_MAPPINGS["Knowledge-Base"].copy()
    MODEL_MAPPINGS["Urchade-PII"] = MODEL_MAPPINGS["E3-JSI-Domains"].copy()

    # ========================================================================
    # HIERARCHICAL LABEL FAMILIES (for relaxed matching)
    # ========================================================================
    # Labels in the same family are considered a match
    # Uses canonical labels (test data format)

    LABEL_FAMILIES = {
        "NAME": {
            "full name"
        },
        "DATE_TIME": {
            "date"
        },
        "LOCATION": {
            "address", "ip address"
        },
        "CONTACT": {
            "phone number", "fax number", "email address"
        },
        "GOVERNMENT_ID": {
            "social security number", "tax identification number",
            "driver's license number", "passport number", "identification number"
        },
        "FINANCIAL_CARD": {
            "credit card number", "credit score"
        },
        "FINANCIAL_ACCOUNT": {
            "bank account number", "iban", "amount"
        },
        "INSURANCE": {
            "insurance number"
        },
        "MEDICAL": {
            "medical condition", "medication", "medical treatment"
        },
        "ORGANIZATION": {
            "organization"
        },
        "ONLINE": {
            "username"
        }
    }

    def __init__(self):
        """Initialize normalizer by building reverse mappings"""
        self._build_reverse_mapping()

    def _build_reverse_mapping(self):
        """Build reverse mapping: variant -> canonical label"""
        self._variant_to_canonical = {}
        
        for canonical, variants in self.CANONICAL_LABELS.items():
            for variant in variants:
                self._variant_to_canonical[variant.lower()] = canonical

    def normalize(self, label: str, model_name: Optional[str] = None) -> str:
        """
        Normalize a label to its canonical form.
        
        Args:
            label: Raw label from model or dataset
            model_name: Optional model name for model-specific mapping
            
        Returns:
            Canonical label name
        """
        label_clean = label.lower().strip()
        
        # Try model-specific mapping first
        if model_name and model_name in self.MODEL_MAPPINGS:
            if label_clean in self.MODEL_MAPPINGS[model_name]:
                return self.MODEL_MAPPINGS[model_name][label_clean]
        
        # Try canonical variant mapping
        if label_clean in self._variant_to_canonical:
            return self._variant_to_canonical[label_clean]
        
        # Return original if no mapping found
        return label_clean

    def labels_match(self, label1: str, label2: str) -> bool:
        """
        Check if two canonical labels match (exact or hierarchical).
        
        Args:
            label1: First canonical label
            label2: Second canonical label
            
        Returns:
            True if labels match exactly or are in same family
        """
        # Exact match
        if label1 == label2:
            return True
        
        # Check hierarchical families
        for family_labels in self.LABEL_FAMILIES.values():
            if label1 in family_labels and label2 in family_labels:
                return True
        
        return False

    def get_all_canonical_labels(self) -> List[str]:
        """Get list of all canonical label names"""
        return list(self.CANONICAL_LABELS.keys())

    def get_label_variants(self, canonical_label: str) -> List[str]:
        """Get all variants for a canonical label"""
        return self.CANONICAL_LABELS.get(canonical_label, [canonical_label])

    def get_model_output_labels(self, model_name: str) -> List[str]:
        """Get expected output labels for a model"""
        if model_name in self.MODEL_MAPPINGS:
            # Return unique canonical labels this model can produce
            return list(set(self.MODEL_MAPPINGS[model_name].values()))
        return []

    def save_reference(self, output_file: str = 'testing/label_normalizer_reference.json'):
        """Save normalization reference for documentation"""
        output = {
            'canonical_labels': self.CANONICAL_LABELS,
            'model_mappings': self.MODEL_MAPPINGS,
            'label_families': {k: list(v) for k, v in self.LABEL_FAMILIES.items()},
            'total_canonical_labels': len(self.CANONICAL_LABELS),
            'total_variants': len(self._variant_to_canonical),
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved normalization reference to {output_file}")


# ============================================================================
# GLOBAL INSTANCE - Use this everywhere
# ============================================================================
normalizer = LabelNormalizer()


# ============================================================================
# CONVENIENCE FUNCTIONS - For backwards compatibility
# ============================================================================

def normalize_label(label: str, model_name: Optional[str] = None) -> str:
    """Normalize a label to canonical form"""
    return normalizer.normalize(label, model_name)


def labels_match(label1: str, label2: str) -> bool:
    """Check if two labels match"""
    return normalizer.labels_match(label1, label2)


def get_canonical_labels() -> List[str]:
    """Get all canonical labels"""
    return normalizer.get_all_canonical_labels()


if __name__ == "__main__":
    print("="*80)
    print("LABEL NORMALIZER - Test with 4 Models")
    print("="*80)

    # Show canonical labels (test data format)
    print("\n=== CANONICAL LABELS (23 test data labels) ===")
    for label in sorted(normalizer.CANONICAL_LABELS.keys()):
        print(f"  - {label}")

    # Test normalizations for each model
    print("\n=== MODEL LABEL NORMALIZATION ===")

    model_tests = {
        "Finetuned-PII": [
            "full name", "date", "social security number", "drivers license number",
            "identity card number", "health insurance id number", "medical treatment"
        ],
        "E3-JSI-Domains": [
            "person", "date of birth", "social security number", "driver's license",
            "identity card", "health insurance ID", "medication", "email"
        ],
        "Gretel-Base": [
            "first_name", "last_name", "date_of_birth", "ssn", "phone_number",
            "credit_card_number", "health_plan_beneficiary_number", "company_name"
        ],
        "Knowledge-Base": [
            "name", "first name", "date of birth", "SSN", "phone number",
            "credit card", "healthcare number", "condition", "medical process"
        ],
    }

    for model_name, labels in model_tests.items():
        print(f"\n{model_name}:")
        for label in labels:
            normalized = normalizer.normalize(label, model_name)
            match_symbol = "✓" if normalized in normalizer.CANONICAL_LABELS else "?"
            print(f"  {match_symbol} {label:<35} -> {normalized}")

    # Test hierarchical matching (using canonical labels)
    print("\n=== HIERARCHICAL MATCHING ===")
    test_matches = [
        ("full name", "full name"),              # Same - should match
        ("phone number", "fax number"),          # CONTACT family - should match
        ("social security number", "passport number"),  # GOVERNMENT_ID - should match
        ("credit card number", "credit score"),  # FINANCIAL_CARD - should match
        ("medical condition", "medication"),     # MEDICAL - should match
        ("phone number", "email address"),       # CONTACT - should match
        ("address", "ip address"),               # LOCATION - should match
        ("full name", "organization"),           # Different families - NO match
    ]

    for label1, label2 in test_matches:
        match = normalizer.labels_match(label1, label2)
        status = "✓ MATCH" if match else "✗ NO MATCH"
        print(f"  {label1:<25} <-> {label2:<25} : {status}")

    print(f"\n=== SUMMARY ===")
    print(f"Canonical labels: {len(normalizer.CANONICAL_LABELS)}")
    print(f"Label variants mapped: {len(normalizer._variant_to_canonical)}")
    print(f"Models configured: {len(normalizer.MODEL_MAPPINGS)}")

    print("\n" + "="*80)
