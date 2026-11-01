
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"
TIMEOUT = 300


# --------------------------------------------------
# OLLAMA REQUEST
# --------------------------------------------------

def ask_ollama(prompt):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict factual claim extraction "
                    "and verification system. Never invent facts. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 8192
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    content = result.get("message", {}).get("content", "")

    if not content:
        raise ValueError("Empty response from Ollama.")

    return json.loads(content)


# --------------------------------------------------
# STAGE 1: GENERATE MULTIPLE CANDIDATE CLAIMS
# --------------------------------------------------

def generate_candidates(passage):

    prompt = f"""
You are an expert factual claim extraction system.

Extract ALL factual propositions from the source passage.

For EACH factual proposition:

1. Create one atomic factual proposition.
2. Generate THREE alternative claim formulations.
3. All three alternatives must express the SAME proposition.
4. Preserve the complete factual meaning.
5. Make each claim independently understandable.
6. Split compound sentences into separate propositions.
7. Do not combine unrelated facts.
8. Preserve exact entities and their roles.
9. Preserve numbers, percentages and measurements.
10. Preserve dates, years and temporal relationships.
11. Preserve subject-action-object relationships.
12. Preserve negation and its scope.
13. Preserve attribution, causality, modality and conditions.
14. Preserve comparisons, degree and geographical scope.
15. Never introduce names, numbers or facts absent from the source.
16. Never use placeholders such as [Name].
17. Do not use external knowledge.
18. Keep the original language.
19. Provide exact source evidence for every proposition.
20. Do not omit any meaningful factual proposition.

IMPORTANT:
Three alternatives are different formulations of ONE fact.
They must not introduce different facts or change its meaning.

SOURCE PASSAGE:
{passage}

Return ONLY valid JSON:

{{
    "propositions": [
        {{
            "source_evidence": "Exact text from the passage",
            "candidates": [
                "First formulation of the claim.",
                "Second formulation of the claim.",
                "Third formulation of the claim."
            ]
        }}
    ]
}}
"""

    return ask_ollama(prompt)


# --------------------------------------------------
# STAGE 2: SELECT THE MOST FACTUALLY CORRECT CLAIM
# --------------------------------------------------

def select_best_claims(passage, candidate_data):

    propositions = candidate_data.get("propositions", [])

    if not propositions:
        return []

    valid_propositions = []

    for proposition in propositions:

        evidence = str(
            proposition.get("source_evidence", "")
        ).strip()

        candidates = proposition.get("candidates", [])

        # Reject evidence not found in the original passage.
        if evidence not in passage:
            continue

        if not isinstance(candidates, list):
            continue

        candidates = [
            str(candidate).strip()
            for candidate in candidates
            if str(candidate).strip()
        ]

        if not candidates:
            continue

        valid_propositions.append({
            "source_evidence": evidence,
            "candidates": candidates
        })

    if not valid_propositions:
        return []

    prompt = f"""
You are an independent factuality evaluator.

Your task is to select the MOST FACTUALLY FAITHFUL
claim from each candidate group.

ORIGINAL SOURCE PASSAGE:
{passage}

CANDIDATE GROUPS:
{json.dumps(valid_propositions, ensure_ascii=False, indent=2)}

EVALUATION RULES:

For every group, evaluate all candidate claims against
the ORIGINAL SOURCE PASSAGE.

Select the candidate that:

1. Preserves the exact meaning of the source.
2. Contains no unsupported information.
3. Preserves all important entities and their roles.
4. Preserves numerical details exactly.
5. Preserves dates and temporal relationships.
6. Preserves subject-predicate-object relationships.
7. Preserves negation and its scope.
8. Preserves attribution and uncertainty.
9. Does not introduce unsupported causality.
10. Does not exaggerate or weaken the original statement.
11. Is atomic and independently understandable.
12. Retains important qualifiers and geographical scope.

REJECTION RULES:

Reject a candidate if it:
- Introduces a person or organization not mentioned.
- Changes a number, date, location or entity.
- Changes who performed an action.
- Changes the relationship between entities.
- Converts an association into causation.
- Changes a negative statement into a positive one.
- Adds assumptions or external information.
- Removes an important qualifier.
- Changes the certainty of the original statement.

SELECTION:

- Select exactly ONE candidate per proposition.
- If multiple candidates are equally accurate, choose
  the clearest and most precise formulation.
- If no candidate is supported, omit that proposition.
- Do not create new facts.
- Do not merge separate propositions.
- Do not remove distinct factual propositions.

Return ONLY valid JSON:

{{
    "claims": [
        "Selected factual claim.",
        "Another selected factual claim."
    ]
}}
"""

    result = ask_ollama(prompt)

    claims = result.get("claims", [])

    if not isinstance(claims, list):
        return []

    return [
        str(claim).strip()
        for claim in claims
        if isinstance(claim, str) and claim.strip()
    ]


# --------------------------------------------------
# COMPLETE CLAIM EXTRACTION PIPELINE
# --------------------------------------------------

def extract_claims(passage):

    if not passage or not passage.strip():
        return []

    # Stage 1: Generate multiple formulations.
    candidates = generate_candidates(passage)

    # Stage 2: Evaluate and select the best formulation.
    selected_claims = select_best_claims(
        passage,
        candidates
    )

    return selected_claims


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    passage = input("Input: ").strip()

    if not passage:
        return

    try:

        claims = extract_claims(passage)

        print("\nClaims found:\n")

        for index, claim in enumerate(claims, 1):
            print(f"{index}. {claim}")

    except requests.RequestException:
        print("Claim extraction failed.")

    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        print("Claim extraction failed.")

    except Exception:
        print("Claim extraction failed.")


if __name__ == "__main__":
    main()
