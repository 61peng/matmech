You are given two texts: **Source Text (the original paper content)** and **Extracted Evidence (the model-generated evidence)**.
Your goal is to determine whether the **Extracted Evidence is fully grounded in the Source Text**, without introducing any unsupported information.

**Evaluation criteria:**

1. If the Extracted Evidence is directly stated in the Source Text or can be _strictly_ inferred from it through minimal logical implication, then it is considered **REAL** (not a hallucination).
2. If the Extracted Evidence adds explanations, assumptions, causal claims, numerical values, definitions, or any detail that **does not appear in the Source Text**, then it is considered **A HALLUCINATION**.
3. Do **not** use external knowledge or domain knowledge. Only judge based on what is present in the Source Text.
4. Provide a confidence score between 0 and 1, where:
   - **1** = completely confident that the Extracted Evidence is REAL (not a hallucination)
   - **0.5** = unclear or partially supported
   - **0** = completely confident that the Extracted Evidence is a HALLUCINATION

**Output strictly in JSON with the following rules:**

- `"consistency": true` means the evidence is REAL (not a hallucination).
- `"consistency": false` means the evidence IS a hallucination.
- `"confidence"` is the probability that the evidence is REAL (not hallucination). Values > 0.5 indicate REAL; values ≤ 0.5 indicate HALLUCINATION.

Output format:
{{"consistency": true/false, "confidence": number}}

**Now evaluate the following pair:**

Source Text (original paper content):
{premise}

Extracted Evidence (model output):
{hypothesis}
