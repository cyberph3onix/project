"""
End-to-end pipeline for backstory consistency verification.

Flow:
1. Load novel text
2. Parse backstory into claims
3. For each claim:
   - Retrieve relevant evidence from novel
   - Evaluate claim consistency
4. Output: 1 (all claims supported) or 0 (any claim fails)
"""

import json
from typing import List
from retrieval.retrieve import chunk_text, retrieve_evidence
from reasoning.evaluate import evaluate_claim


# ========== SAMPLE DATA ==========
SAMPLE_NOVEL = """
The kingdom of Eldoria had stood for three centuries. King Marcus ruled with wisdom and justice.
He was known for his patience and his love of ancient texts. The royal library contained thousands of volumes.

The queen, Lady Sophia, was a renowned healer. She had trained under the monks of Mount Avery for five years.
Her knowledge of herbs and medicines saved countless lives during the plague of 1523.

Prince Adrian, their son, was born in 1518. He showed great interest in military strategy from an early age.
His father taught him the art of diplomacy, and by age twenty, he had negotiated three major treaties.

The court historian records that Prince Adrian married Elena of the Southern Provinces in 1540.
Elena brought with her a dowry of three thousand gold coins and a library of scientific texts.
Prince Adrian and Elena had two children: Margaret (born 1541) and Thomas (born 1543).

In 1550, a terrible flood destroyed the eastern settlements. Prince Adrian personally led the relief effort.
He stayed in the flooded regions for six months, organizing rebuilding efforts.

Margaret grew up in the castle and became an expert in languages. By age eighteen, she could speak five languages.
Thomas, her brother, preferred life outside the castle. He became a merchant and traveled extensively.
"""

SAMPLE_BACKSTORY = """
Adrian is the eldest son of King Marcus and Queen Sophia. He became interested in strategy as a child.
He married a woman named Elena from the Southern Provinces in 1540.
Adrian had a deep sense of duty and personally intervened during the great flood of 1550 to help his people.
"""


def parse_backstory_into_claims(backstory: str) -> List[str]:
    """
    Convert backstory text into discrete, verifiable claims.
    
    In production, use NLI or fine-tuned model. For now, use simple heuristics:
    split by sentences and filter for claims (not rhetorical statements).
    
    Args:
        backstory: Free-text backstory description
    
    Returns:
        List of claims to verify
    """
    # Simple: split by period, clean up
    sentences = [s.strip() for s in backstory.split('.') if s.strip()]
    
    # Filter: keep sentences that contain verifiable facts
    # (have entity + predicate structure)
    claims = []
    for sent in sentences:
        # Very basic filter: length > 10 chars and contains a verb-like word
        if len(sent) > 10 and any(verb in sent.lower() for verb in ['is', 'was', 'became', 'had', 'married']):
            claims.append(sent)
    
    return claims


def verify_backstory(novel_text: str, backstory_text: str) -> int:
    """
    Main verification pipeline.
    
    Args:
        novel_text: Full novel text (can be very long)
        backstory_text: Backstory to verify
    
    Returns:
        1 if all backstory claims are supported, 0 otherwise
    """
    
    # Step 1: Prepare evidence (chunk the novel)
    print("[1/4] Chunking novel text...")
    chunks = chunk_text(novel_text, chunk_size=500, overlap=100)
    print(f"     Created {len(chunks)} chunks")
    
    # Step 2: Parse backstory into claims
    print("[2/4] Parsing backstory into claims...")
    claims = parse_backstory_into_claims(backstory_text)
    print(f"     Extracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"     - Claim {i}: {claim[:60]}...")
    
    # Step 3: Evaluate each claim
    print("[3/4] Evaluating claims...")
    results = []
    all_supported = True
    
    for i, claim in enumerate(claims, 1):
        # Retrieve evidence
        evidence = retrieve_evidence(claim, chunks, k=5)
        
        # Evaluate
        verdict = evaluate_claim(claim, evidence)
        results.append({
            "claim": claim,
            "verdict": verdict,
            "evidence_count": len(evidence)
        })
        
        print(f"     Claim {i}: {verdict.upper()}")
        
        # Conservative: any non-support = fail
        if verdict != "support":
            all_supported = False
    
    # Step 4: Aggregate
    print("[4/4] Aggregating results...")
    final_label = 1 if all_supported else 0
    print(f"     Final verdict: {final_label}")
    
    return final_label, results


def save_report(results: List, output_path: str = "report/verification_report.json"):
    """
    Save detailed verification results to JSON.
    
    Args:
        results: List of claim evaluation results
        output_path: Where to save the report
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Report saved to {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("BACKSTORY CONSISTENCY VERIFICATION")
    print("=" * 70)
    
    # Run pipeline
    final_label, results = verify_backstory(SAMPLE_NOVEL, SAMPLE_BACKSTORY)
    
    # Save report
    save_report(results)
    
    # Final output
    print("\n" + "=" * 70)
    print(f"FINAL OUTPUT: {final_label}")
    print("=" * 70)
    print(f"Interpretation: {'CONSISTENT - Backstory is supported' if final_label == 1 else 'CONTRADICTORY - Backstory has inconsistencies'}")
    print("\nDetailed results:")
    for i, res in enumerate(results, 1):
        print(f"  {i}. [{res['verdict'].upper()}] {res['claim'][:50]}...")
