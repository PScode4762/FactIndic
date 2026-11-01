"""FactIndic: four-model, cross-evaluated summary refinement.

Exactly four distinct Ollama models are used. Each generates one candidate;
each candidate is evaluated by the OTHER three models. Claim labels are
aggregated by majority vote and numeric scores by arithmetic mean. The
highest-scoring candidate is selected. No fifth model is required.

Required files in the same directory:
    claim_extractor.py
    evaluator.py

Run:
    python refiner.py
    python refiner.py --language hi
    python refiner.py --bertscore

Paste the source passage and existing summary into the terminal. Finish each
input with END. A separate human/reference summary is optional; enter SKIP
when prompted if you do not have one.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from claim_extractor import extract_claims
import evaluator

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
TIMEOUT = 300

# Exactly four models. Each generates one candidate and cross-evaluates
# the candidates produced by the other three models.
GENERATION_MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
    "gemma3:4b",
    "mistral:7b",
]

GENERATOR_SYSTEM = """
You are a careful factual summarization editor for FactIndic.
The source passage is the only factual authority. Improve the existing
summary while preserving important information supported by the passage.
Correct errors and remove unsupported details. Do not invent entities,
numbers, dates, locations, causes, comparisons, or conclusions. Preserve
negation, attribution, uncertainty, conditions, and qualifiers. Do not turn
correlation into causation. Write a concise, coherent summary in the requested
language. Return only the summary, without a heading or explanation.
""".strip()

EVALUATOR_SYSTEM = """
You are a strict evidence-grounded factuality evaluator. Evaluate the
candidate ONLY against the source passage. The source passage is the ultimate
evidence; extracted claims are prompts, not evidence themselves.

Return one valid JSON object:
{
  "claim_results": [
    {"claim_number": 1, "status": "SUPPORTED", "explanation": "..."}
  ],
  "unsupported_additions": [],
  "contradictions": [],
  "relevance_score": 0,
  "coherence_score": 0,
  "overall_factuality_score": 0,
  "brief_reason": "..."
}

For every extracted source claim, return exactly one status:
SUPPORTED, PARTIALLY_SUPPORTED, OMITTED, UNSUPPORTED, or CONTRADICTED.
SUPPORTED means accurately represented and supported by the passage.
PARTIALLY_SUPPORTED means only part is expressed or supported.
OMITTED means the candidate does not express the source claim; omission is
not hallucination. UNSUPPORTED means the candidate adds a fact not supported
by the passage. CONTRADICTED means the candidate conflicts with the passage.

Scores must be integers from 0 to 10:
- relevance_score: relevance to the passage's main topic.
- coherence_score: clarity, grammar, and logical flow.
- overall_factuality_score: factual accuracy of statements actually made.
Be conservative. Penalize fabricated details, changed numbers/entities,
unsupported causality, and altered certainty. Do not penalize omission as
hallucination; omission is tracked separately. Include every claim exactly
once. Return JSON only, with no Markdown fences or extra commentary.
""".strip()


def ask_ollama(prompt: str, system_prompt: str, model: str,
               temperature: float = 0.2, json_mode: bool = False) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "").strip()
    if not content:
        raise ValueError(f"Ollama model {model!r} returned an empty response.")
    return content


def get_installed_models() -> set:
    response = requests.get(OLLAMA_TAGS_URL, timeout=30)
    response.raise_for_status()
    return {
        item.get("name", "")
        for item in response.json().get("models", [])
        if item.get("name")
    }


def check_required_models(generation_models: Sequence[str]) -> None:
    """Check that the four configured Ollama models are installed."""
    if len(generation_models) != 4 or len(set(generation_models)) != 4:
        raise ValueError("Exactly four distinct generation models are required.")

    installed = get_installed_models()
    missing = [name for name in generation_models if name not in installed]
    if missing:
        commands = "\n".join(f"  ollama pull {name}" for name in missing)
        raise RuntimeError(
            "These four-model ensemble members are not installed:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\n\nInstall them using:\n" + commands
        )


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge model did not return a valid JSON object.")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object.")
    return value


def read_multiline_input(title: str, optional: bool = False) -> Optional[str]:
    print(f"\n{title}")
    if optional:
        print("Enter SKIP on the first line if you do not have one.")
    print("Paste/type the text below. Enter END on a new line when finished:")
    lines: List[str] = []
    first_line = True
    while True:
        try:
            line = input()
        except EOFError:
            break
        if first_line and optional and line.strip().upper() == "SKIP":
            return None
        first_line = False
        if line.strip() == "END":
            break
        lines.append(line)
    value = "\n".join(lines).strip()
    if not value:
        if optional:
            return None
        raise ValueError(f"{title} cannot be empty.")
    return value


def normalize_claims(article: str,
                    supplied_claims: Optional[Sequence[Any]] = None) -> List[str]:
    if supplied_claims is None:
        print("\n[FactIndic] Extracting claims using claim_extractor.py...")
        supplied_claims = extract_claims(article)
    claims: List[str] = []
    for item in supplied_claims:
        if isinstance(item, str):
            claim = item.strip()
        elif isinstance(item, dict):
            claim = str(item.get("claim", item.get("text", ""))).strip()
        else:
            continue
        if claim and claim not in claims:
            claims.append(claim)
    if not claims:
        raise ValueError("No claims were extracted. Check claim_extractor.py output.")
    return claims


def generate_candidate(article: str, original_summary: str,
                       claims: Sequence[str], model: str,
                       candidate_number: int, language: str) -> str:
    claims_text = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, 1))
    language_name = "Hindi" if language == "hi" else "English"
    prompt = f"""
SOURCE PASSAGE:
{article}

EXTRACTED SOURCE CLAIMS:
{claims_text}

EXISTING SUMMARY TO REFINE:
{original_summary}

TASK:
You are generation model candidate {candidate_number}. Produce one refined
summary in {language_name}. Use the source passage as the only factual
authority. Preserve important supported facts and do not invent details.
Return only the summary.
"""
    return ask_ollama(prompt, GENERATOR_SYSTEM, model, temperature=0.2).strip()


def evaluate_candidate(article: str, claims: Sequence[str], candidate: str,
                       evaluator_model: str) -> Dict[str, Any]:
    claims_text = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, 1))
    prompt = f"""
SOURCE PASSAGE:
{article}

EXTRACTED SOURCE CLAIMS (verify against passage; do not assume they are true):
{claims_text}

CANDIDATE SUMMARY:
{candidate}

Assess every source claim and the factuality of statements in the candidate.
Return the JSON structure requested in your instructions.
"""
    raw = ask_ollama(prompt, EVALUATOR_SYSTEM, evaluator_model,
                     temperature=0.0, json_mode=True)
    return parse_json_response(raw)


def normalize_evaluation(evaluation: Dict[str, Any],
                         claims: Sequence[str]) -> Dict[str, Any]:
    allowed = {"SUPPORTED", "PARTIALLY_SUPPORTED", "OMITTED",
               "UNSUPPORTED", "CONTRADICTED"}
    raw_results = evaluation.get("claim_results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    by_number: Dict[int, Dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("claim_number"))
        except (TypeError, ValueError):
            continue
        status = str(item.get("status", "")).strip().upper()
        if 1 <= number <= len(claims) and status in allowed:
            by_number[number] = {
                "claim_number": number,
                "claim": claims[number - 1],
                "status": status,
                "explanation": str(item.get("explanation", "")).strip(),
            }
    normalized = []
    for number, claim in enumerate(claims, 1):
        normalized.append(by_number.get(number, {
            "claim_number": number,
            "claim": claim,
            "status": "OMITTED",
            "explanation": "No valid judge assessment returned; conservatively treated as omitted/unassessed.",
        }))

    def score(key: str) -> int:
        try:
            return max(0, min(10, int(round(float(evaluation.get(key, 0))))) )
        except (TypeError, ValueError):
            return 0

    unsupported = evaluation.get("unsupported_additions", [])
    contradictions = evaluation.get("contradictions", [])
    if not isinstance(unsupported, list):
        unsupported = []
    if not isinstance(contradictions, list):
        contradictions = []
    return {
        "claim_results": normalized,
        "unsupported_additions": unsupported,
        "contradictions": contradictions,
        "relevance_score": score("relevance_score"),
        "coherence_score": score("coherence_score"),
        "overall_factuality_score": score("overall_factuality_score"),
        "brief_reason": str(evaluation.get("brief_reason", "")).strip(),
    }


def aggregate_peer_evaluations(
    evaluations: Sequence[Dict[str, Any]],
    claims: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate three peer-model evaluations of one candidate.

    Claim labels use majority vote. Numeric scores use the arithmetic mean.
    Unsupported additions and contradictions are combined without duplicates.
    """
    if len(evaluations) != 3:
        raise ValueError(
            f"Each candidate must be evaluated by the other three models; "
            f"received {len(evaluations)} evaluations."
        )

    normalized = [normalize_evaluation(item, claims) for item in evaluations]
    label_order = [
        "SUPPORTED", "PARTIALLY_SUPPORTED", "OMITTED",
        "UNSUPPORTED", "CONTRADICTED"
    ]
    conservative_order = [
        "CONTRADICTED", "UNSUPPORTED", "OMITTED",
        "PARTIALLY_SUPPORTED", "SUPPORTED"
    ]

    aggregated_claims = []
    for claim_index, claim in enumerate(claims):
        votes = [item["claim_results"][claim_index]["status"] for item in normalized]
        counts = {label: votes.count(label) for label in label_order}
        majority = max(
            conservative_order,
            key=lambda label: (counts[label], -conservative_order.index(label))
        )
        explanations = [
            item["claim_results"][claim_index]["explanation"]
            for item in normalized
            if item["claim_results"][claim_index]["status"] == majority
            and item["claim_results"][claim_index]["explanation"]
        ]
        aggregated_claims.append({
            "claim_number": claim_index + 1,
            "claim": claim,
            "status": majority,
            "votes": counts,
            "explanation": explanations[0] if explanations else
                "Aggregated from three peer-model evaluations.",
        })

    def mean_score(key: str) -> int:
        return int(round(sum(item[key] for item in normalized) / len(normalized)))

    def unique_items(key: str) -> List[Any]:
        result: List[Any] = []
        seen = set()
        for item in normalized:
            for value in item[key]:
                marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if marker not in seen:
                    seen.add(marker)
                    result.append(value)
        return result

    return {
        "claim_results": aggregated_claims,
        "unsupported_additions": unique_items("unsupported_additions"),
        "contradictions": unique_items("contradictions"),
        "relevance_score": mean_score("relevance_score"),
        "coherence_score": mean_score("coherence_score"),
        "overall_factuality_score": mean_score("overall_factuality_score"),
        "brief_reason": "Peer-evaluation aggregate. " + " | ".join(
            item["brief_reason"] for item in normalized if item["brief_reason"]
        ),
        "peer_evaluations": normalized,
        "aggregation_method": {
            "claim_labels": "majority vote across the other three models",
            "numeric_scores": "arithmetic mean across the other three models",
            "unsupported_additions_and_contradictions": "deduplicated union",
        },
    }


def factuality_metrics_for_candidate(normalized: Dict[str, Any]) -> Dict[str, Any]:
    all_results = normalized["claim_results"]
    non_omitted = [
        {"label": item["status"]}
        for item in all_results if item["status"] != "OMITTED"
    ]
    # evaluator.py supports SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED,
    # CONTRADICTED. OMITTED is separately tracked and not counted as a hallucination.
    factuality = evaluator.evaluate_factuality(non_omitted)
    counts = {label: 0 for label in
              ("SUPPORTED", "PARTIALLY_SUPPORTED", "OMITTED", "UNSUPPORTED", "CONTRADICTED")}
    for item in all_results:
        counts[item["status"]] += 1
    total = len(all_results)
    expressed = total - counts["OMITTED"]
    weighted = (counts["SUPPORTED"] + 0.5 * counts["PARTIALLY_SUPPORTED"])
    return {
        "evaluator_factuality": factuality,
        "source_claim_counts": counts,
        "total_source_claims": total,
        "expressed_claims": expressed,
        "omission_rate": counts["OMITTED"] / total if total else None,
        "claim_coverage_rate": expressed / total if total else None,
        "weighted_claim_coverage_rate": weighted / total if total else None,
    }


def evaluate_with_project_evaluator(candidate: str, language: str,
                                    normalized: Dict[str, Any],
                                    reference_summary: Optional[str] = None,
                                    include_bertscore: bool = False,
                                    bertscore_model_type: Optional[str] = None) -> Dict[str, Any]:
    claim_metrics = factuality_metrics_for_candidate(normalized)
    non_omitted = [
        {"label": item["status"]}
        for item in normalized["claim_results"] if item["status"] != "OMITTED"
    ]
    result: Dict[str, Any] = {
        "summary_quality": None,
        "factuality": claim_metrics,
        "evidence_retrieval": None,
        "reference_summary_used": reference_summary is not None,
    }
    if reference_summary:
        combined = evaluator.evaluate_all(
            reference=reference_summary,
            generated=candidate,
            claim_results=non_omitted,
            language=language,
            include_bertscore=include_bertscore,
            bertscore_model_type=bertscore_model_type,
        )
        result["summary_quality"] = combined["summary_quality"]
        result["evaluator_factuality"] = combined["factuality"]
    return result


def calculate_candidate_score(evaluation: Dict[str, Any],
                              project_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a transparent 0-100 score from common judge/evaluator metrics.

    Score = 40% judge factuality + 25% weighted claim coverage +
            10% relevance + 10% coherence + 10% (1 - hallucination rate) +
             5% (1 - contradiction rate).

    ROUGE/BLEU/BERTScore are reported when a reference is provided but are not
    included in this score, so candidates remain comparable when no reference
    summary is available.
    """
    fact = project_metrics["factuality"]["evaluator_factuality"]
    hallucination = fact.get("hallucination_rate")
    contradiction = fact.get("contradiction_rate")
    # If no claims were assessed, penalize rather than treating missing metrics as zero risk.
    hallucination = 1.0 if hallucination is None else float(hallucination)
    contradiction = 1.0 if contradiction is None else float(contradiction)
    coverage = project_metrics["factuality"]["weighted_claim_coverage_rate"]
    coverage = 0.0 if coverage is None else float(coverage)

    components = {
        "factuality_points_40": 4.0 * evaluation["overall_factuality_score"],
        "weighted_claim_coverage_points_25": 25.0 * coverage,
        "relevance_points_10": evaluation["relevance_score"],
        "coherence_points_10": evaluation["coherence_score"],
        "low_hallucination_points_10": 10.0 * (1.0 - hallucination),
        "low_contradiction_points_5": 5.0 * (1.0 - contradiction),
    }
    total = round(sum(components.values()), 2)
    return {"score_out_of_100": total, "score_components": components,
            "scoring_formula": "40% factuality + 25% weighted claim coverage + 10% relevance + 10% coherence + 10% hallucination avoidance + 5% contradiction avoidance"}


def refine_summary(article: str, summary: str,
                   claims: Optional[Sequence[Any]] = None,
                   generation_models: Optional[Sequence[str]] = None,
                   language: str = "en",
                   reference_summary: Optional[str] = None,
                   include_bertscore: bool = False,
                   bertscore_model_type: Optional[str] = None,
                   check_models: bool = True) -> Dict[str, Any]:
    """Generate four candidates and cross-evaluate each with the other models."""
    article = evaluator.validate_text(article, "source passage")
    summary = evaluator.validate_text(summary, "summary to refine")
    language = language.strip().lower()
    evaluator.validate_language(language)
    models = list(generation_models or GENERATION_MODELS)
    if len(models) != 4 or len(set(models)) != 4:
        raise ValueError("Exactly four distinct generation models must be provided.")
    if reference_summary is not None:
        reference_summary = evaluator.validate_text(reference_summary, "reference summary")
    if check_models:
        check_required_models(models)

    selected_claims = normalize_claims(article, claims)
    candidates: List[Dict[str, Any]] = []

    for index, generation_model in enumerate(models, 1):
        print(f"\n[FactIndic] Model {index}/4 generating candidate: {generation_model}")
        candidate_text = generate_candidate(
            article, summary, selected_claims, generation_model, index, language
        )
        if not candidate_text:
            print(f"[Warning] {generation_model} returned an empty candidate; skipping.")
            continue

        peer_models = [model for model in models if model != generation_model]
        peer_evaluations = []
        for peer_model in peer_models:
            print(
                f"[FactIndic] {peer_model} evaluating candidate {index} "
                f"(generated by {generation_model})..."
            )
            raw_eval = evaluate_candidate(
                article, selected_claims, candidate_text, peer_model
            )
            peer_evaluations.append(raw_eval)

        aggregated = aggregate_peer_evaluations(peer_evaluations, selected_claims)
        project_metrics = evaluate_with_project_evaluator(
            candidate_text, language, aggregated, reference_summary,
            include_bertscore, bertscore_model_type
        )
        score = calculate_candidate_score(aggregated, project_metrics)
        record = {
            "candidate_number": index,
            "generation_model": generation_model,
            "evaluated_by_models": peer_models,
            "summary": candidate_text,
            "evaluation": aggregated,
            "project_metrics": project_metrics,
            "selection_score": score,
        }
        candidates.append(record)
        print(
            f"[FactIndic] {generation_model}: "
            f"{score['score_out_of_100']}/100 "
            f"(mean of 3 peer evaluations)"
        )

    if not candidates:
        raise RuntimeError("No candidate was successfully generated and evaluated.")

    candidates.sort(key=lambda item: (
        item["selection_score"]["score_out_of_100"],
        -item["candidate_number"]
    ), reverse=True)
    best = candidates[0]

    return {
        "language": language,
        "generation_models": models,
        "evaluation_method": (
            "Each of the four models generates one candidate. Each candidate "
            "is evaluated by the other three models; claim labels are aggregated "
            "by majority vote and numeric scores by arithmetic mean."
        ),
        "original_summary": summary,
        "reference_summary_provided": reference_summary is not None,
        "claims": selected_claims,
        "best_summary": best["summary"],
        "best_candidate_number": best["candidate_number"],
        "best_generation_model": best["generation_model"],
        "best_score": best["selection_score"]["score_out_of_100"],
        "best_score_components": best["selection_score"]["score_components"],
        "selection_method": best["selection_score"]["scoring_formula"],
        "best_evaluation": best["evaluation"],
        "best_project_metrics": best["project_metrics"],
        "number_of_candidates_generated": len(candidates),
        "candidates": candidates,
        "metric_notes": {
            "models": "Exactly four Ollama models are used; no fifth model is required.",
            "peer_evaluation": "Each candidate is judged by the other three models, never by its own generator model.",
            "aggregation": "Claim labels use majority vote; numeric evaluation scores use the mean of the three peer judgments.",
            "reference_metrics": "ROUGE, BLEU and optional BERTScore are calculated only when a separate human/reference summary is supplied; they are not part of the selection score.",
            "omitted_claims": "Omitted claims are tracked separately and are not counted as hallucinations.",
            "recall_at_k": "Unavailable until actual relevant/retrieved evidence IDs are connected from the retriever.",
            "research_caution": "Peer LLM judgments are model-based estimates, not ground truth. Manually audit a sample of claims and report agreement where possible.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four summaries with four Ollama models; each candidate "
            "is evaluated by the other three models. No fifth model is used."
        )
    )
    parser.add_argument("--language", choices=["en", "hi"], default="en")
    parser.add_argument(
        "--models",
        default=",".join(GENERATION_MODELS),
        help="Exactly four comma-separated distinct Ollama model tags.",
    )
    parser.add_argument("--output", default="refined_summary.txt")
    parser.add_argument("--report", default="refinement_report.json")
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Calculate optional BERTScore when a reference summary is supplied.",
    )
    parser.add_argument(
        "--bertscore-model",
        default=None,
        help="Optional multilingual/Hindi-capable Hugging Face checkpoint.",
    )
    args = parser.parse_args()

    try:
        models = [model.strip() for model in args.models.split(",") if model.strip()]
        if len(models) != 4 or len(set(models)) != 4:
            raise ValueError(
                "--models must contain exactly four distinct comma-separated model tags."
            )

        passage = read_multiline_input("Enter the source passage")
        original_summary = read_multiline_input("Enter the existing summary to refine")
        reference_summary = read_multiline_input(
            "Enter a human/reference summary (optional)", optional=True
        )
        if reference_summary is None:
            print(
                "\n[FactIndic] No reference summary supplied; "
                "ROUGE/BLEU/BERTScore will be unavailable."
            )

        result = refine_summary(
            article=passage,
            summary=original_summary,
            generation_models=models,
            language=args.language,
            reference_summary=reference_summary,
            include_bertscore=args.bertscore,
            bertscore_model_type=args.bertscore_model,
        )

        output_path = Path(args.output)
        report_path = Path(args.report)
        output_path.write_text(result["best_summary"] + "\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 68)
        print("FACTINDIC - BEST FOUR-MODEL REFINED SUMMARY")
        print("=" * 68)
        print(result["best_summary"])
        print(f"\nWinning model: {result['best_generation_model']}")
        print(f"Selection score: {result['best_score']}/100")
        print("\nCandidate ranking:")
        for rank, item in enumerate(result["candidates"], 1):
            print(
                f"  {rank}. {item['generation_model']}: "
                f"{item['selection_score']['score_out_of_100']}/100"
            )
        print(f"\nSaved summary: {output_path.resolve()}")
        print(f"Evaluation report: {report_path.resolve()}")

    except requests.ConnectionError:
        print(
            "\n[ERROR] Cannot connect to Ollama. Start Ollama and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.HTTPError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else ""
        print(f"\n[ERROR] Ollama HTTP error: {exc}\n{detail}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"\n[ERROR] Ollama request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
