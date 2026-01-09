"""
Reasoning module for claim evaluation.

Core logic:
- Given a claim and retrieved evidence chunks
- Decide: does evidence SUPPORT, CONTRADICT, or remain UNCLEAR?
- Conservative: when in doubt, treat as contradiction (fail-safe)
"""

from typing import List, Literal


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
    
    # ========== PLACEHOLDER LLM CALL ==========
    # In production, replace with actual LLM (e.g., OpenAI API):
    #
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": _build_system_prompt()
    #         },
    #         {
    #             "role": "user",
    #             "content": _build_user_prompt(claim, evidence_chunks)
    #         }
    #     ],
    #     temperature=0.0
    # )
    # verdict = response["choices"][0]["message"]["content"].strip().lower()
    
    # For now, use heuristic fallback:
    verdict = _heuristic_evaluate(claim_lower, combined_evidence, evidence_chunks)
    
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