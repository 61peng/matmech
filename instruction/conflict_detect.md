Evaluate whether the following scientific statements contain **semantic or factual conflicts**.
The source statement describes scientific knowledge extracted from a materials science paper.
Each candidate statement is semantically similar (cosine similarity > 0.9) and may potentially conflict with the source.

---

### **Source Statement**

{source_sentence}

### **Candidate Statements**

{candidate_list}
(Each item has an ID for reference)

---

### **Task**

For each candidate statement, determine whether it **conflicts** with the source statement.
A conflict means that the candidate contradicts, negates, reverses, or imposes a mutually exclusive condition relative to the source.

Examples of conflicts:

- Opposite causal relations
- Opposite experimental results
- Mutually exclusive material behaviors
- Inconsistent numerical values or trends
- Logical contradictions in mechanism explanation

---

### **Output Format (strict JSON)**

Return a **JSON array**, one object per candidate, following exactly this schema:

```
[
  {{
    "candidate_id": "",
    "conflict": true/false,
    "reason": "Short explanation of why it conflicts or not."
  }},
  ...
]
```

Requirements:

- Use only JSON as output (no additional text).
- If conflict, explain why they are compatible in `"reason"`.
- If no conflict, set `"reason": "none"`.
