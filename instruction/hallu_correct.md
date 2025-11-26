Your task is to determine whether a given claim is supported by any block within a full scientific paper represented as a JSON list.

You are provided with:

1. The full paper as an array of JSON objects.
2. A claim (viewpoint).
3. A block index that has already been checked and is known either:
   (a) to be insufficient on its own to support the claim,
   OR
   (b) to be excluded entirely because it is irrelevant or misleading for evaluating the claim.

   — In case (a), the block may still contain context, but it must NOT be treated as sufficient evidence.
   — In case (b), you must ignore this block completely and not use it in any form.
   — In both cases, you must search the rest of the document to determine whether any other block supports the claim.

---

## Your Required Reasoning Rules

---

1. Search ALL blocks in the paper EXCEPT the provided excluded block.
2. A block supports the claim only if the meaning of the block _directly and explicitly_ entails the claim.
   - Paraphrase matches are allowed.
   - Inferences must remain strictly grounded in the text.
   - Do NOT hallucinate missing experimental methods, conclusions, or interpretations.
3. If you find at least one block that supports the claim:
   • Output "supported": true  
    • Output the supporting block index  
    • Do NOT output hallucination_type
4. If NO block supports the claim:
   • Output "supported": false  
    • Output hallucination_type as one of: - "factual errors"
   - "logical contradictions"
   - "fabrications"
     • Choose the hallucination category that best describes why the claim cannot be derived from any part of the paper.
5. Output must be valid JSON.
6. Do NOT include any explanation outside the JSON.

---

## JSON Output Format

---

```json
{{
"supported": true/false,
"block_index": number or null,
"hallucination_type": "factual errors" | "logical contradictions" | "fabrications" | null
}}
```

---

## Inputs

---

Full Paper JSON List:
{paper_json}

Claim:
"{claim}"

Checked Block Index:
{excluded_block}

---

## Task

---

Now perform the document-level verification following the rules above.
