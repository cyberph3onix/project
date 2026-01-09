"""
Reasoning module for claim evaluation.

Core logic:
- Given a claim and retrieved evidence chunks
- Decide: does evidence SUPPORT, CONTRADICT, or remain UNCLEAR?
- Conservative: when in doubt, treat as contradiction (fail-safe)
"""

from typing import List, Literal
import os
import json

# Optional LLM integration (OpenAI). If no API key, fall back to heuristic.
_openai_available = False
try:
    import openai
    _openai_available = True
except Exception:
    _openai_available = False


def evaluate_claim(claim: str, evidence_chunks: List[str]) -> Literal["support", "contradict", "unclear"]:
    """
    Evaluate whether evidence supports or contradicts a claim.
    
    Implementation strategy:
    1. In production: call an LLM (e.g., GPT-4) with a structured prompt
    2. Here: simulate with rule-based heuristics + placeholder for LLM
    
    Conservative logic:
    - If evidence strongly mentions claim content → "support"
    - If evidence contradicts claim → "contradict"
    - If evidence is sparse or ambiguous → "unclear" (treat as fail)
    
    Args:
        claim: The factual claim to evaluate
        evidence_chunks: Retrieved text chunks from the novel
    
    Returns:
        One of: "support", "contradict", "unclear"
    """
    
    # Combine all evidence
    combined_evidence = " ".join(evidence_chunks).lower()
    claim_lower = claim.lower()
    
    # Attempt to use an LLM judge. The LLM must return exactly one of three labels:
    #   - "contradiction"
    #   - "no_contradiction"
    #   - "insufficient_evidence"
    # We then map those to the pipeline's internal labels:
    #   no_contradiction -> support
    #   contradiction -> contradict
    #   insufficient_evidence -> unclear
    llm_label = None

    # Build minimal user prompt (claim + evidence only)
    evidence_text = "\n---\n".join(evidence_chunks) if evidence_chunks else "[NO EVIDENCE]"
    user_prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence excerpts:\n{evidence_text}\n\n"
        "Please output EXACTLY ONE of these three words (only the word, no explanation):\n"
        "contradiction\nno_contradiction\ninsufficient_evidence"
    )

    # Use OpenAI if available and API key present
    try:
        if _openai_available and os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            # Use chat completion; deterministic with temperature=0
            resp = openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a concise binary judge. Reply with one exact label only."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            raw = resp["choices"][0]["message"]["content"].strip()
            llm_label = raw.splitlines()[0].strip().lower()
    except Exception:
        llm_label = None

    # If LLM not available or returned unexpected output, fallback to heuristic
    if llm_label not in {"contradiction", "no_contradiction", "insufficient_evidence"}:
        mapped = _heuristic_evaluate(claim_lower, combined_evidence, evidence_chunks)
        return mapped

    # Map LLM labels to internal verdicts used by the rest of the pipeline
    mapping = {
        "contradiction": "contradict",
        "no_contradiction": "support",
        "insufficient_evidence": "unclear",
    }
    verdict = mapping[llm_label]

    # Optional: request a 2-3 sentence rationale and save to rationales.json
    # Only do this if OpenAI key present (avoid extra API calls otherwise)
    try:
        if _openai_available and os.environ.get("OPENAI_API_KEY"):
            rationale_prompt = (
                f"Claim: {claim}\n\nEvidence:\n{evidence_text}\n\n"
                f"Label: {llm_label}\n\n"
                "Provide a 2-3 sentence rationale explaining why this label was chosen."
            )
            resp2 = openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a concise explainer. Provide exactly 2-3 short sentences."},
                    {"role": "user", "content": rationale_prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            rationale = resp2["choices"][0]["message"]["content"].strip()
            # Save rationale entry
            try:
                out_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "rationales.json")
                entry = {"claim": claim, "label": llm_label, "rationale": rationale}
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as rf:
                        data = json.load(rf)
                else:
                    data = []
                data.append(entry)
                with open(out_path, "w", encoding="utf-8") as wf:
                    json.dump(data, wf, indent=2, ensure_ascii=False)
            except Exception:
                pass
    except Exception:
        pass

    return verdict


def _heuristic_evaluate(claim_lower: str, combined_evidence: str, evidence_chunks: List[str]) -> str:
    """
    Rule-based fallback when LLM is unavailable.
    
    This is a placeholder and will give naive results.
    Replace with actual LLM reasoning in production.
    """
    
    # Extract key entities/keywords from claim
    claim_words = set(claim_lower.split())
    claim_words = {w for w in claim_words if len(w) > 3}  # Filter short words
    
    # Check if key claim words appear in evidence
    matches = sum(1 for word in claim_words if word in combined_evidence)
    coverage = matches / len(claim_words) if claim_words else 0
    
    # Naive thresholds
    if coverage >= 0.6:
        return "support"  # Most key terms found
    elif coverage < 0.2 and len(evidence_chunks) > 0:
        # Evidence exists but doesn't match → potential contradiction
        return "unclear"  # Conservative: treat unclear as fail
    else:
        return "unclear"  # Ambiguous; conservative default


def _build_system_prompt() -> str:
    """
    System prompt for the LLM.
    Instructs the model how to evaluate claims.
    """
    return """You are a literary analysis expert. Your task is to evaluate whether a claim about a character's backstory is consistent with a novel.

Instructions:
1. Read the claim carefully.
2. Examine the provided evidence excerpts from the novel.
3. Decide if the evidence SUPPORTS the claim, CONTRADICTS it, or is UNCLEAR.

Rules:
- SUPPORT: Evidence explicitly or strongly implies the claim is true.
- CONTRADICT: Evidence explicitly or strongly implies the claim is false.
- UNCLEAR: Evidence is missing, ambiguous, or insufficient to decide.

Output exactly one word: "support", "contradict", or "unclear"."""


def _build_user_prompt(claim: str, evidence_chunks: List[str]) -> str:
    """
    User prompt for the LLM.
    Provides the claim and evidence.
    """
    evidence_text = "\n---\n".join(evidence_chunks) if evidence_chunks else "[NO EVIDENCE FOUND]"
    
    return f"""Claim: {claim}

Evidence from the novel:
{evidence_text}

Verdict (support/contradict/unclear):"""