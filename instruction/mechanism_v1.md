You are given:

1. A **materials science paper**, formatted as a **JSON list**, where each item represents a paragraph or figure from the paper.  
2. A **core knowledge summary**, which outlines the key tetrahedral concepts and logical relationships (example shown below).

Your task is to extract the **core scientific mechanism** based on the **materials tetrahedron framework** and express it in structured JSON format.

---

## Step-by-Step Guide


### Step 1: Read the Full Text JSON

You will be given the full paper as a JSON list. The structure of each dictionary is as follows:  

- **"type"**: Indicates the block type, which can be `"text"`, `"image"`, or `"table"`.  
- **"text"**: Present when `"type"` is `"text"`, containing the paragraph content.  
- **"img_path"**: Present when `"type"` is `"image"` or `"table"`, containing the file path of the figure or table.  
- **"microscopic_image"**: Present when `"type"` is `"image"`, determine whether the image is a microscopic image.  
- **"function"**: Present when `"type"` is `"image"`, a string representing the image's role.  
- **"block_index"**: The unique identifier for each block.  
- **"text_level"**: (Optional) Indicates the heading level if applicable.  


---

### Step 2: Understand the Core Knowledge Summary

You will be provided a JSON object summarizing the key elements of the paper:

- `"material object"`: The subject of the study.  
- `"Processing"`: The applied process.  
- `"Structure"`: The resulting structural changes.  
- `"Properties"`: The influenced properties.  
- `"Performance"`: The performance outcome.  
- `"logical chain"`: The causal sequence linking these elements (e.g., `"Processing → Structure → Property"`).  

This summary tells you:
- What material is studied
- What each tetrahedral concept refers to
- How these concepts are logically linked

---

### Step 3: Extract All Actual Cause-Effect Links Described in the Paper

Cause-Effect Links can be the relationship between any two elements in logical chain, or even the same element, only if described in the text.

**Do not infer links not supported in the paper.**

---

### Step 4: For Each Cause-Effect Link, Output a JSON Object with the Following Fields

```json
[{
  "link": "Processing → Structure", 
  "Cause": "Processing description (from paper)",
  "effect": "Structure description (from paper)",
  "description": "One-sentence summary explaining how the cause influences the effect",
  "experiment": {
    "name": "Name of the experiment",
    "parameters": "Important settings like temperature, duration, load, etc.",
    "result": "What was observed or measured"
    "source": [block_index] 
  },
  "images": [
    {
      "block_index": 14,
      "image description": "The text describing the image in the paper, which can be used to explain the mechanism",
      "source": [block_index]
    }
  ],
  "referenced_knowledge": [
    {
      "content": "Knowledge or findings from other sources cited in the paper that can served as evidence for explaining the mechanism",
      "reference": ["[2] H. Yang, J. C. Love, F. Arias, G. M. Whitesides, Chem. Mater. 2002, 14, 1385."],
      "source": [block_index]
    }
  ],
  "domain_knowledge": [
    {
      "content": "Common principles or known facts in materials science that are relevant to the mechanism",
      "source": [block_index]
    }
  ],
  "mechanism": {
    "description": "Explanation of how the cause leads to the effect, including physical/logical processes",
    "source": [block_index],
    "reasoning_chain": [{
      "statement": "One step in the logical explanation",
      "type": "Choose from: experimental result, image description, domain knowledge, referenced knowledge, deductive reasoning, inductive reasoning"
      }, ...]
  }
}, ...]
```

---

### Step 5: Construct a Complete, Coherent Reasoning Chain

- The `reasoning_chain` must reflect the **step-by-step logical flow from the cause to the effect**.
- It must **start from the cause**, go through intermediate steps (experimental/visual/textual/logical), and **end at the effect**.
- Each reasoning step must be tagged using one of:
  - `experimental result`
  - `image description`
  - `domain knowledge`
  - `referenced knowledge`

- all extracted `experimental results`, `image descriptions`, `domain knowledge`, and `referenced knowledge` should be included in the `reasoning_chain`.
- Add `deductive reasoning`, `inductive reasoning` steps as needed to make the chain fluent and coherent.

This ensures the **reasoning_chain is complete and comprehensive**, not fragmentary.

---

### Final Notes

- **Use only information explicitly stated** in the paper.
- **Every field must include a `"source"`** (block_index list) to indicate origin.
- **All extracted experimental result, image description, domain knowledge and referenced knowledge must be included in the reasoning chain**.
- The `"reference"` field in `referenced_knowledge` must **match the actual references** listed at the end of the paper.
- Output must be a clean **JSON array**, with each identified link as a separate object.