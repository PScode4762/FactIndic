"""
FactIndic: Evaluation Module
============================

Evaluates English and Hindi summaries using:

1. ROUGE-1, ROUGE-2, ROUGE-L
2. BLEU
3. Optional BERTScore
4. Claim-level factuality metrics
5. Evidence retrieval Recall@K
6. Combined evaluation report
7. Unit tests and demonstration

Supported languages:
    en -> English
    hi -> Hindi

Important:
- ROUGE and BLEU measure textual similarity, not factual correctness.
- Claim-level factuality metrics aggregate labels provided by a verifier.
- Recall@K evaluates evidence retrieval when relevant evidence IDs are known.
- BERTScore is optional and requires the bert-score package.
- The demo uses illustrative data, not actual FactIndic experiment results.

Use python evaluator.py --demo --language en for English
Use python evaluator.py --demo --language hi for Hindi
"""

import argparse
import re
import unittest
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ============================================================
# 1. CONFIGURATION
# ============================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
}

VALID_LABELS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
}


# ============================================================
# 2. VALIDATION AND TOKENIZATION
# ============================================================

def validate_language(language: str) -> str:
    """
    Validate a language code.

    Returns:
        The full language name.

    Raises:
        ValueError: If the language is not English or Hindi.
    """
    if not isinstance(language, str):
        raise ValueError("Language must be a string: 'en' or 'hi'.")

    language = language.strip().lower()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            "Supported languages are 'en' (English) and 'hi' (Hindi)."
        )

    return SUPPORTED_LANGUAGES[language]


def validate_text(text: str, name: str = "text") -> str:
    """Validate and normalize a text input."""
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a string.")

    text = text.strip()

    if not text:
        raise ValueError(f"{name} cannot be empty.")

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Unicode-aware tokenization for English and Hindi.

    Preserves Devanagari combining marks so words such as
    'भारत', 'देश' and 'है' remain intact.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    # Match letters, combining marks, and optional apostrophes.
    # \p{M} is supported by the third-party 'regex' package.
    try:
        import regex
        pattern = r"[^\W\d_](?:[\p{M}\w'’]*)"
        return regex.findall(pattern, text, flags=regex.UNICODE)
    except ImportError:
        # Fallback using Python's standard library.
        # Include Unicode combining marks explicitly.
        import unicodedata

        tokens = []
        current = []

        for char in text:
            category = unicodedata.category(char)

            if char.isalpha() or category.startswith("M"):
                current.append(char)
            elif char in ("'", "’") and current:
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []

        if current:
            tokens.append("".join(current))

        return tokens


# ============================================================
# 3. ROUGE
# ============================================================

def _get_rouge_scorer():
    """Create a ROUGE scorer if rouge-score is installed."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )


def calculate_rouge(
    reference: str,
    generated: str,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L.

    Returns precision, recall, and F1 for each metric.
    If rouge-score is unavailable, returns None values.
    """
    reference = validate_text(reference, "reference")
    generated = validate_text(generated, "generated")

    scorer = _get_rouge_scorer()

    metric_names = ["rouge1", "rouge2", "rougeL"]

    if scorer is None:
        return {
            metric: None
            for metric in metric_names
        }

    scores = scorer.score(reference, generated)

    return {
        metric: {
            "precision": float(scores[metric].precision),
            "recall": float(scores[metric].recall),
            "f1": float(scores[metric].fmeasure),
        }
        for metric in metric_names
    }


# ============================================================
# 4. BLEU
# ============================================================

def calculate_bleu(
    reference: str,
    generated: str,
) -> Optional[float]:
    """
    Calculate sentence-level BLEU with smoothing.

    Returns:
        BLEU score between 0 and 1, or None if NLTK is unavailable.
    """
    reference = validate_text(reference, "reference")
    generated = validate_text(generated, "generated")

    try:
        from nltk.translate.bleu_score import (
            sentence_bleu,
            SmoothingFunction,
        )
    except ImportError:
        return None

    reference_tokens = tokenize_text(reference)
    generated_tokens = tokenize_text(generated)

    if not reference_tokens or not generated_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1

    score = sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=smoothing,
    )

    return float(score)


# ============================================================
# 5. BERTSCORE (OPTIONAL)
# ============================================================

def calculate_bertscore(
    reference: str,
    generated: str,
    lang: str = "en",
    model_type: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Calculate BERTScore precision, recall, and F1.

    Args:
        reference: Reference summary.
        generated: Generated summary.
        lang: 'en' or 'hi'.
        model_type: Optional Hugging Face model checkpoint.

    Notes:
        The 'lang' parameter alone does not guarantee that BERTScore
        selects a suitable Hindi model. For Hindi, supply an appropriate
        multilingual or Hindi-capable checkpoint through model_type.

    Returns:
        Dictionary containing precision, recall, and F1, or None if
        the bert-score package is unavailable.
    """
    reference = validate_text(reference, "reference")
    generated = validate_text(generated, "generated")
    validate_language(lang)

    try:
        from bert_score import score as bert_score
    except ImportError:
        return None

    kwargs = {
        "cands": [generated],
        "refs": [reference],
        "lang": lang,
        "verbose": False,
    }

    if model_type:
        kwargs["model_type"] = model_type

    precision, recall, f1 = bert_score(**kwargs)

    return {
        "precision": float(precision[0].item()),
        "recall": float(recall[0].item()),
        "f1": float(f1[0].item()),
    }


# ============================================================
# 6. SUMMARY QUALITY EVALUATION
# ============================================================

def evaluate_summary_quality(
    reference: str,
    generated: str,
    include_bertscore: bool = True,
    bertscore_lang: str = "en",
    bertscore_model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a generated summary against a reference summary.

    BERTScore is optional. Its result is None when disabled or when
    the bert-score package is unavailable.
    """
    reference = validate_text(reference, "reference")
    generated = validate_text(generated, "generated")
    validate_language(bertscore_lang)

    rouge_results = calculate_rouge(reference, generated)
    bleu_result = calculate_bleu(reference, generated)

    bertscore_result = None

    if include_bertscore:
        bertscore_result = calculate_bertscore(
            reference=reference,
            generated=generated,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
        )

    return {
        "rouge": rouge_results,
        "bleu": bleu_result,
        "bertscore": bertscore_result,
        "reference_word_count": len(tokenize_text(reference)),
        "generated_word_count": len(tokenize_text(generated)),
    }


# ============================================================
# 7. CLAIM-LEVEL FACTUALITY EVALUATION
# ============================================================

def evaluate_factuality(
    claim_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate claim-level verification results.

    Each item must contain a 'label' field with one of:
        SUPPORTED
        PARTIALLY_SUPPORTED
        UNSUPPORTED
        CONTRADICTED

    This function does not verify claims itself. It summarizes the
    labels supplied by the project's verifier.
    """
    if isinstance(claim_results, (str, bytes)):
        raise TypeError("claim_results must be a sequence of dictionaries.")

    if not claim_results:
        return {
            "total_claims": 0,
            "label_counts": {label: 0 for label in sorted(VALID_LABELS)},
            "support_rate": None,
            "partial_support_rate": None,
            "hallucination_rate": None,
            "contradiction_rate": None,
        }

    counts = Counter()

    for index, result in enumerate(claim_results):
        if not isinstance(result, dict):
            raise TypeError(
                f"Claim result at index {index} must be a dictionary."
            )

        if "label" not in result:
            raise ValueError(
                f"Claim result at index {index} is missing 'label'."
            )

        label = result["label"]

        if not isinstance(label, str):
            raise ValueError(
                f"Claim label at index {index} must be a string."
            )

        label = label.strip().upper()

        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid claim label {label!r}. "
                f"Allowed labels: {sorted(VALID_LABELS)}"
            )

        counts[label] += 1

    total = len(claim_results)

    supported = counts["SUPPORTED"]
    partially_supported = counts["PARTIALLY_SUPPORTED"]
    unsupported = counts["UNSUPPORTED"]
    contradicted = counts["CONTRADICTED"]

    return {
        "total_claims": total,
        "label_counts": {
            label: counts[label]
            for label in sorted(VALID_LABELS)
        },
        "support_rate": supported / total,
        "partial_support_rate": partially_supported / total,
        "hallucination_rate": (unsupported + contradicted) / total,
        "contradiction_rate": contradicted / total,
    }


# ============================================================
# 8. EVIDENCE RETRIEVAL: RECALL@K
# ============================================================

def calculate_recall_at_k(
    relevant_evidence_ids: Sequence[Any],
    retrieved_evidence_ids: Sequence[Any],
    k: int = 5,
) -> Optional[float]:
    """
    Calculate Recall@K.

    Recall@K =
        relevant evidence IDs found in the first K retrieved IDs
        ---------------------------------------------------------
                   total relevant evidence IDs

    Returns None when there are no relevant evidence IDs.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    relevant = set(relevant_evidence_ids)

    if not relevant:
        return None

    retrieved_top_k = list(retrieved_evidence_ids)[:k]
    retrieved_set = set(retrieved_top_k)

    relevant_retrieved = relevant.intersection(retrieved_set)

    return len(relevant_retrieved) / len(relevant)


# ============================================================
# 9. COMBINED EVALUATION
# ============================================================

def evaluate_all(
    reference: str,
    generated: str,
    claim_results: Optional[Sequence[Dict[str, Any]]] = None,
    relevant_evidence_ids: Optional[Sequence[Any]] = None,
    retrieved_evidence_ids: Optional[Sequence[Any]] = None,
    k: int = 5,
    language: str = "en",
    include_bertscore: bool = True,
    bertscore_model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the available evaluation components and return one report.

    Supported languages are English and Hindi only.
    """
    validate_language(language)

    summary_metrics = evaluate_summary_quality(
        reference=reference,
        generated=generated,
        include_bertscore=include_bertscore,
        bertscore_lang=language,
        bertscore_model_type=bertscore_model_type,
    )

    factuality_metrics = None
    retrieval_metrics = None

    if claim_results is not None:
        factuality_metrics = evaluate_factuality(claim_results)

    if (
        relevant_evidence_ids is not None
        and retrieved_evidence_ids is not None
    ):
        retrieval_metrics = {
            f"recall@{k}": calculate_recall_at_k(
                relevant_evidence_ids=relevant_evidence_ids,
                retrieved_evidence_ids=retrieved_evidence_ids,
                k=k,
            )
        }

    return {
        "language": language,
        "language_name": validate_language(language),
        "summary_quality": summary_metrics,
        "factuality": factuality_metrics,
        "evidence_retrieval": retrieval_metrics,
    }


# ============================================================
# 10. REPORT PRINTING
# ============================================================

def print_report(report: Dict[str, Any]) -> None:
    """Print the evaluation report in a readable format."""
    print("\n" + "=" * 60)
    print("FACTINDIC EVALUATION REPORT")
    print("=" * 60)

    print(f"Language: {report['language_name']} ({report['language']})")

    summary = report["summary_quality"]

    print("\n--- Summary Quality ---")

    print("\nROUGE:")
    for metric, values in summary["rouge"].items():
        if values is None:
            print(f"  {metric.upper()}: unavailable (install rouge-score)")
        else:
            print(
                f"  {metric.upper()}: "
                f"P={values['precision']:.4f}, "
                f"R={values['recall']:.4f}, "
                f"F1={values['f1']:.4f}"
            )

    bleu = summary["bleu"]

    if bleu is None:
        print("BLEU: unavailable (install nltk)")
    else:
        print(f"BLEU: {bleu:.4f}")

    bertscore = summary["bertscore"]

    if bertscore is None:
        print("BERTScore: not calculated or unavailable")
    else:
        print(
            "BERTScore: "
            f"P={bertscore['precision']:.4f}, "
            f"R={bertscore['recall']:.4f}, "
            f"F1={bertscore['f1']:.4f}"
        )

    print(
        f"Reference word count: "
        f"{summary['reference_word_count']}"
    )
    print(
        f"Generated word count: "
        f"{summary['generated_word_count']}"
    )

    factuality = report["factuality"]

    if factuality is not None:
        print("\n--- Claim-Level Factuality ---")
        print(f"Total claims: {factuality['total_claims']}")

        for label, count in factuality["label_counts"].items():
            print(f"{label}: {count}")

        for metric in (
            "support_rate",
            "partial_support_rate",
            "hallucination_rate",
            "contradiction_rate",
        ):
            value = factuality[metric]

            if value is None:
                print(f"{metric}: N/A")
            else:
                print(f"{metric}: {value:.4f}")

    retrieval = report["evidence_retrieval"]

    if retrieval is not None:
        print("\n--- Evidence Retrieval ---")

        for metric, value in retrieval.items():
            if value is None:
                print(f"{metric}: N/A (no relevant evidence IDs)")
            else:
                print(f"{metric}: {value:.4f}")

    print("\n" + "=" * 60)


# ============================================================
# 11. DEMO
# ============================================================

def run_demo(language: str = "en") -> None:
    """
    Run a small illustrative evaluation example.

    The claim labels and evidence IDs below are manually supplied
    demonstration data. They are not actual FactIndic experiment results.
    """
    language_name = validate_language(language)

    if language == "en":
        reference = (
            "The government announced a new education policy "
            "to improve rural schools."
        )
        generated = (
            "The government announced a new policy "
            "to improve rural schools."
        )
    else:
        reference = (
            "सरकार ने ग्रामीण स्कूलों में सुधार के लिए "
            "नई शिक्षा नीति की घोषणा की।"
        )
        generated = (
            "सरकार ने ग्रामीण स्कूलों में सुधार के लिए "
            "नई नीति की घोषणा की।"
        )

    # Illustrative labels only. Replace with actual verifier output
    # when integrating this module into the FactIndic pipeline.
    claim_results = [
        {"claim": "A new policy was announced.", "label": "SUPPORTED"},
        {"claim": "The policy has already succeeded.", "label": "UNSUPPORTED"},
        {"claim": "The policy was cancelled.", "label": "CONTRADICTED"},
    ]

    # Illustrative evidence IDs only. Replace with IDs from your retriever.
    relevant_evidence_ids = ["evidence_1", "evidence_2"]
    retrieved_evidence_ids = [
        "evidence_1",
        "evidence_3",
        "evidence_4",
    ]

    print(f"Running illustrative {language_name} demo...")

    report = evaluate_all(
        reference=reference,
        generated=generated,
        claim_results=claim_results,
        relevant_evidence_ids=relevant_evidence_ids,
        retrieved_evidence_ids=retrieved_evidence_ids,
        k=2,
        language=language,
        include_bertscore=False,
    )

    print_report(report)

    print(
        "\nNote: These are demonstration results, not measured "
        "FactIndic experimental results."
    )


# ============================================================
# 12. UNIT TESTS
# ============================================================

class TestEvaluator(unittest.TestCase):

    def test_supported_languages(self):
        self.assertEqual(validate_language("en"), "English")
        self.assertEqual(validate_language("hi"), "Hindi")

    def test_language_case_normalization(self):
        self.assertEqual(validate_language("EN"), "English")
        self.assertEqual(validate_language("HI"), "Hindi")

    def test_unsupported_language(self):
        with self.assertRaises(ValueError):
            validate_language("pa")

    def test_tokenize_english(self):
        tokens = tokenize_text("FactIndic evaluates factual summaries.")
        self.assertEqual(
            tokens,
            ["FactIndic", "evaluates", "factual", "summaries"],
        )

    def test_tokenize_hindi(self):
        tokens = tokenize_text("भारत एक देश है।")
        self.assertTrue(tokens)
        self.assertIn("भारत", tokens)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            validate_text("   ")

    def test_factuality_rates(self):
        claims = [
            {"label": "SUPPORTED"},
            {"label": "PARTIALLY_SUPPORTED"},
            {"label": "UNSUPPORTED"},
            {"label": "CONTRADICTED"},
        ]

        result = evaluate_factuality(claims)

        self.assertEqual(result["total_claims"], 4)
        self.assertAlmostEqual(result["support_rate"], 0.25)
        self.assertAlmostEqual(result["partial_support_rate"], 0.25)
        self.assertAlmostEqual(result["hallucination_rate"], 0.50)
        self.assertAlmostEqual(result["contradiction_rate"], 0.25)

    def test_empty_claims(self):
        result = evaluate_factuality([])

        self.assertEqual(result["total_claims"], 0)
        self.assertIsNone(result["support_rate"])
        self.assertIsNone(result["hallucination_rate"])

    def test_invalid_claim_label(self):
        with self.assertRaises(ValueError):
            evaluate_factuality([{"label": "UNKNOWN"}])

    def test_recall_at_k(self):
        score = calculate_recall_at_k(
            relevant_evidence_ids=["a", "b"],
            retrieved_evidence_ids=["a", "c", "b"],
            k=2,
        )

        self.assertAlmostEqual(score, 0.5)

    def test_recall_no_relevant_evidence(self):
        score = calculate_recall_at_k([], ["a", "b"], k=2)
        self.assertIsNone(score)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            calculate_recall_at_k(["a"], ["a"], k=0)

    def test_summary_quality(self):
        result = evaluate_summary_quality(
            reference="The cat sat on the mat.",
            generated="The cat sat on the mat.",
            include_bertscore=False,
        )

        self.assertIn("rouge", result)
        self.assertIn("bleu", result)
        self.assertIsNone(result["bertscore"])
        self.assertEqual(result["reference_word_count"], 6)
        self.assertEqual(result["generated_word_count"], 6)

    def test_combined_report(self):
        report = evaluate_all(
            reference="भारत एक देश है।",
            generated="भारत एक देश है।",
            claim_results=[{"label": "SUPPORTED"}],
            relevant_evidence_ids=["e1"],
            retrieved_evidence_ids=["e1"],
            k=1,
            language="hi",
            include_bertscore=False,
        )

        self.assertEqual(report["language"], "hi")
        self.assertEqual(report["language_name"], "Hindi")
        self.assertEqual(
            report["factuality"]["support_rate"],
            1.0,
        )
        self.assertEqual(
            report["evidence_retrieval"]["recall@1"],
            1.0,
        )


def run_tests() -> bool:
    """Run the evaluator's unit tests."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestEvaluator
    )

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    return result.wasSuccessful()


# ============================================================
# 13. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FactIndic evaluator for English and Hindi summarization."
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the illustrative evaluation demo.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the evaluator unit tests.",
    )

    parser.add_argument(
        "--language",
        choices=["en", "hi"],
        default="en",
        help="Language for the demo: en (English) or hi (Hindi).",
    )

    args = parser.parse_args()

    if args.test:
        success = run_tests()

        if not success:
            raise SystemExit(1)

    elif args.demo:
        run_demo(language=args.language)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
