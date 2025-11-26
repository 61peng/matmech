## Task Definition

You are given:

1. A **materials science paper**, formatted as a JSON list. Each item represents a paragraph or figure.
2. A **core knowledge summary** that outlines key tetrahedral concepts:

   * Material Object
   * Processing
   * Structure
   * Property
   * Performance
   * Logical Chain: the ideal causal pathway (e.g., `"Processing → Structure → Property → Performance"`)

Your task is to extract all **explicitly described scientific mechanisms** from the paper and represent them in a structured JSON format following the materials tetrahedron framework.

---

## Input Format

### 1. Full Paper (`full_paper`)

A list of JSON objects with the following fields:

```json
{
  "type": "text" | "image" | "table" | ...,
  "block_index": 5,
  "text": "...",               // for "text"
  "img_path": "...",          // for "image"
  "microscopic_image": true,  // only for image
  "function": "...",          // only for image
  "text_level": 1             // optional
}
```

### 2. Core Knowledge Summary (`core_summary`)

```json
{
  "material object": "SnTe and Pb1-xSnxSe(Te)",
  "Processing": "...",
  "Structure": "...",
  "Property": "...",
  "Performance": "...",
  "logical chain": "Processing → Structure → Property → Performance"
}
```

---

## Output Format

Your output must be a JSON array. Each object in the array represents a single cause-effect link explicitly described in the paper:

```json
[
  { /* First mechanism */
    "link": "Processing → Structure",
    "cause": "...",   // Taken from paper
    "effect": "...",  // Taken from paper
    "description": "Summarize in one sentence how cause leads to effect",
    "experiment": {
      "name": "...",
      "type": "e.g., TEM, XRD, tensile test",
      "parameters": "...",
      "result": "...",
      "source": [block_index]
    },
    "images": [
      {
        "block_index": 14, // The position of the image itself
        "image description": "...",
        "microscopic": true,
        "source": [block_index]  // The postion of image description
      }
    ],
    "referenced_knowledge": [
      {
        "content": "...",
        "reference": ["[3] J. Mater. Sci. 2020, 55, 1234."],
        "source": [block_index]
      }
    ],
    "domain_knowledge": [
      {
        "content": "...",
        "source": [block_index]
      }
    ],
    "mechanism": {
      "description": "Scientific explanation of how the cause leads to the effect",
      "source": [block_index],
      "reasoning_chain": [
        {
          "statement": "...",
          "type": "experimental result | image description | domain knowledge | referenced knowledge | deductive reasoning | inductive reasoning"
        }
      ]
    }
  },
  { /* Second mechanism */ },
  ...
]
```

---

## Extraction Rules

1. Only extract links **explicitly described** in the paper text or captions.
2. You may extract multiple links per paper (e.g., `Processing → Structure`, `Structure → Property`, etc.).
3. Every field must include a `"source"` list pointing to the `block_index` in the full paper.
4. The `"reasoning_chain"` must include all elements mentioned in `"experiment"`, `"images"`, `"referenced_knowledge"`, and `"domain_knowledge"`, and must start from the cause and end at the effect.
5. Do **not** infer mechanisms unless directly supported.

---

## Reasoning Types (for each reasoning\_chain step)

* `"experimental result"`: From a named experiment in the paper
* `"image description"`: Textual explanation associated with a figure
* `"domain knowledge"`: General materials science principle stated in the paper
* `"referenced knowledge"`: From a cited source
* `"deductive reasoning"`: Logical deduction (e.g., symmetry breaking leads to band gap)
* `"inductive reasoning"`: Inference based on observed pattern or trend

---

## Example Output

```json
[
  {
    "link": "Processing → Structure",
    "cause": "Molecular beam epitaxy growth along the (001) direction",
    "effect": "SnTe and Pb1-xSnxSe(Te) films exhibit spin-filtered edge states protected by (001) mirror symmetry",
    "description": "Epitaxial growth along (001) leads to thin films with mirror-symmetry-protected topological edge states.",
    "experiment": {
      "name": "TCI film band structure computation",
      "type": "Tight-binding & k·p analysis",
      "parameters": "Growth on (001), various thicknesses (3, 5, 25 layers)",
      "result": "5-layer film exhibits band inversion with mirror symmetry protection",
      "source": [22]
    },
    "images": [
      {
        "block_index": 25,
        "image description": "Band inversion observed in SnTe films of different thicknesses, confirming topological phase emergence.",
        "microscopic": false,
        "source": [22]
      }
    ],
    "referenced_knowledge": [
      {
        "content": "SnTe has mirror-symmetry-protected surface states",
        "reference": ["[7] Hsieh, T. H. et al., Nature Commun. 3, 982 (2012)."],
        "source": [3]
      }
    ],
    "domain_knowledge": [
      {
        "content": "Growth direction in epitaxial films influences symmetry and resulting electronic topology.",
        "source": [8]
      }
    ],
    "mechanism": {
      "description": "Growth along (001) preserves mirror symmetry, enabling formation of topologically non-trivial states at the film edges.",
      "source": [8],
      "reasoning_chain": [
        {
          "statement": "SnTe films are typically grown by molecular beam epitaxy in a layer-by-layer mode.",
          "type": "experimental result"
        },
        {
          "statement": "Figure shows thickness-dependent band inversion, with topological edge states present in thicker films.",
          "type": "image description"
        },
        {
          "statement": "SnTe exhibits mirror-symmetry-protected surface states.",
          "type": "referenced knowledge"
        },
        {
          "statement": "Growth orientation preserves or breaks symmetry depending on direction and thickness.",
          "type": "domain knowledge"
        },
        {
          "statement": "Thus, (001) growth yields mirror symmetry that supports edge states with non-trivial topology.",
          "type": "deductive reasoning"
        }
      ]
    }
  }
]
```

---

## Common Pitfalls to Avoid

| Mistake                                        | Fix                                                  |
| ---------------------------------------------- | ---------------------------------------------------- |
| Inferring unstated mechanisms                  | Only extract explicitly described cause-effect links |
| Missing `source` fields                        | Every claim must point to a `block_index`            |
| Reasoning chain too short                      | Include all evidence + logical steps                 |
| Repeating the same sentence in multiple fields | Use distinct, appropriate wording per field          |

---

## Final Reminder

Your output must be a **clean JSON array**, one object per mechanism link, and each must contain:

* Complete fields
* Verifiable source
* Coherent reasoning chain