**Task:**
Given a research paper represented as a JSON list of content blocks (each with fields such as `"type"`, `"text"`, `"text_level"`, `"page_idx"`), determine the **type of study** based on whether the paper investigates a _specific material or material system_ and whether it explores the **Processing–Structure–Property–Performance (PSPP)** relationships that form the Materials Tetrahedron.

---

### **Classification Criteria**

| Category                                                | Label                            | Definition                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Type A – Four-Element Relationship Study**            | `"FourElementRelationshipStudy"` | The paper explicitly investigates causal links among processing, structure, property, and/or performance of a specific material or material system. It quantitatively or qualitatively relates how processing affects structure, how structure determines properties, or how these govern performance. |
| **Type B – Single-Aspect Material Study**               | `"SingleAspectMaterialStudy"`    | The paper focuses on a particular material or class of materials, but discusses only one or two aspects (e.g., synthesis or microstructure) without connecting them across the PSPP chain.                                                                                                             |
| **Type C – Device / System / Method Study**             | `"DeviceOrMethodStudy"`          | The main subject is a device, process, or measurement/modeling method, rather than intrinsic material relationships.                                                                                                                                                                                   |
| **Type D – Theoretical / Computational / Review Study** | `"TheoreticalOrReviewStudy"`     | The paper is primarily theoretical, computational, or a review; it does not center on new experimental investigation of a material system.                                                                                                                                                             |
| **Type E – Non-Research Article**                       | `"NonResearchArticle"`           | The document is not a research article (e.g., Corrigendum, Editorial, Cover Image, News, Perspective, Announcement, or Front Matter).                                                                                                                                                                  |

---

### **Expected Output Format (JSON)**

```json
{
  "classification": "<CategoryLabel>",
  "reasoning": "<Concise academic explanation (1–2 sentences)>",
  "detected_material_system": "<If applicable, specify the material or system studied, otherwise null>"
}
```

---

### **Example Input**

```json
[
  {
    "type": "text",
    "text": "Effect of annealing temperature on microstructure and mechanical properties of Ti–6Al–4V alloy",
    "text_level": 1,
    "page_idx": 0
  },
  {
    "type": "text",
    "text": "This study investigates how heat treatment temperature influences grain morphology, dislocation density, and tensile strength of Ti–6Al–4V.",
    "page_idx": 0
  }
]
```

### **Example Output**

```json
{
  "classification": "FourElementRelationshipStudy",
  "reasoning": "The paper explicitly connects processing (annealing temperature) to structure (grain morphology) and property (tensile strength) of a specific alloy.",
  "detected_material_system": "Ti–6Al–4V alloy"
}
```
