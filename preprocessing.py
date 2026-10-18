
import re
import unicodedata
import unittest


# =========================================================
# PREPROCESSING FUNCTIONS
# =========================================================

def validate_record(record):
    """Validate article and summary fields."""
    errors = []

    for field in ("article", "summary"):
        if field not in record:
            errors.append(f"Missing field: {field}")
        elif not isinstance(record[field], str):
            errors.append(f"{field} must be a string")
        elif not record[field].strip():
            errors.append(f"{field} is empty")

    return errors


def normalize_text(text):
    """Conservatively normalize text without changing its meaning."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    text = unicodedata.normalize("NFC", text)

    # Remove control/format characters except common whitespace
    text = "".join(
        char for char in text
        if unicodedata.category(char) not in {"Cc", "Cf"}
        or char in "\n\r\t"
    )

    # Normalize line endings and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_language(text, detector=None):
    """Use an optional language detector; never guess if absent."""
    text = normalize_text(text)

    if not text:
        return {"language": "unknown", "code_mixed": False}

    if detector is None:
        return {"language": "unknown", "code_mixed": False}

    result = detector(text)

    if isinstance(result, str):
        language, code_mixed = result, False
    elif isinstance(result, dict):
        language = result.get("language", "unknown")
        code_mixed = result.get("code_mixed", False)
    elif isinstance(result, (tuple, list)) and result:
        language = result[0]
        code_mixed = result[1] if len(result) > 1 else False
    else:
        raise ValueError("Invalid detector output")

    return {
        "language": language,
        "code_mixed": bool(code_mixed)
    }


def clean_records(records, deduplicate=True):
    """Clean valid records and report rejected records."""
    accepted, rejected = [], []
    seen = set()

    for index, record in enumerate(records):
        errors = validate_record(record)

        if errors:
            rejected.append({"index": index, "errors": errors})
            continue

        cleaned = dict(record)

        for field in ("article", "summary"):
            cleaned[field + "_original"] = record[field]
            cleaned[field] = normalize_text(record[field])

        key = cleaned["article"].casefold()

        if deduplicate and key in seen:
            rejected.append({
                "index": index,
                "errors": ["Duplicate article"]
            })
            continue

        seen.add(key)
        accepted.append(cleaned)

    return accepted, rejected


def prepare_model_input(text, tokenizer, max_length=512):
    """Prepare text using a compatible pretrained-model tokenizer."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    return tokenizer(
        normalize_text(text),
        max_length=max_length,
        truncation=True
    )


# =========================================================
# TEST CASES
# Run: python test_preprocessing.py
# =========================================================

class TestPreprocessing(unittest.TestCase):

    def test_valid_record(self):
        record = {
            "article": "Delhi hosted the event in 2025.",
            "summary": "Delhi hosted the event."
        }
        self.assertEqual(validate_record(record), [])

    def test_missing_summary(self):
        self.assertTrue(
            validate_record({"article": "An article"})
        )

    def test_empty_article(self):
        self.assertTrue(
            validate_record({
                "article": "   ",
                "summary": "A summary"
            })
        )

    def test_wrong_data_type(self):
        self.assertTrue(
            validate_record({
                "article": 123,
                "summary": "A summary"
            })
        )

    def test_whitespace_normalization(self):
        result = normalize_text("  Hello   world \n test ")
        self.assertEqual(result, "Hello world test")

    def test_unicode_normalization(self):
        result = normalize_text("Cafe\u0301")
        self.assertEqual(result, "Café")
        self.assertTrue(unicodedata.is_normalized("NFC", result))

    def test_preserves_numbers_dates_and_negation(self):
        text = "NOT approved! Amount: ₹5,000 in 2025."
        result = normalize_text(text)
        self.assertEqual(result, text)

    def test_preserves_hindi(self):
        text = "सरकार ने योजना शुरू नहीं की।"
        self.assertEqual(normalize_text(text), text)

    def test_preserves_hinglish(self):
        text = "Sarkar ne scheme launch nahi ki in 2025."
        self.assertEqual(normalize_text(text), text)

    def test_rejects_non_string_input(self):
        with self.assertRaises(TypeError):
            normalize_text(123)

    def test_language_unknown_without_detector(self):
        result = detect_language("नमस्ते")
        self.assertEqual(result["language"], "unknown")

    def test_language_detection_with_detector(self):
        def fake_detector(text):
            return {"language": "hi-en", "code_mixed": True}

        result = detect_language(
            "Yeh ek test hai", detector=fake_detector
        )
        self.assertEqual(result["language"], "hi-en")
        self.assertTrue(result["code_mixed"])

    def test_clean_records_preserves_original(self):
        records = [{
            "article": "  Delhi   hosted the event. ",
            "summary": "The event happened."
        }]

        accepted, rejected = clean_records(records)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(
            accepted[0]["article"], "Delhi hosted the event."
        )
        self.assertEqual(
            accepted[0]["article_original"],
            "  Delhi   hosted the event. "
        )

    def test_rejects_invalid_records(self):
        records = [
            {"article": "No summary"},
            {"article": " ", "summary": "Some summary"}
        ]

        accepted, rejected = clean_records(records)

        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 2)

    def test_removes_duplicate_articles(self):
        records = [
            {"article": "Delhi hosted an event.",
             "summary": "Summary one"},
            {"article": " delhi hosted an event. ",
             "summary": "Summary two"}
        ]

        accepted, rejected = clean_records(records)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)

    def test_tokenizer_integration(self):
        def fake_tokenizer(text, **kwargs):
            return {"text": text, **kwargs}

        result = prepare_model_input(
            "  Hello   world ",
            fake_tokenizer,
            max_length=64
        )

        self.assertEqual(result["text"], "Hello world")
        self.assertEqual(result["max_length"], 64)
        self.assertTrue(result["truncation"])

    def test_invalid_max_length(self):
        with self.assertRaises(ValueError):
            prepare_model_input("Hello", lambda text, **kw: {}, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
