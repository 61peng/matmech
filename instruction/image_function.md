I will give you a list of dictionaries representing blocks from material science papers, where each dictionary corresponds to a paragraph, figure, or table. The structure of each dictionary is as follows:  

- **"type"**: Indicates the block type, which can be `"text"`, `"image"`, or `"table"`.  
- **"text"**: Present when `"type"` is `"text"`, containing the paragraph content.  
- **"img_path"**: Present when `"type"` is `"image"`, containing the file path of the figure.  
- **"block_index"**: The unique identifier for each block.  
- **"text_level"**: (Optional) Indicates the heading level if applicable.  

### Task:  

For each block where `"type"` is `"image"`, determine:  

1. **Whether the image is a microscopic image**:  
   - Output `"microscopic_image": true` if the image is a microscopic image (e.g., SEM, TEM, optical microscopy).  
   - Otherwise, output `"microscopic_image": false`.

2. **The function of the image**: Categorize its role as one of the following:  
   - **Characterizing X**: If the image is used to visually describe or analyze a specific aspect of the study.  
   - **Revealing the influence from X to Y**: If the image is used to illustrate how one factor (X) affects another factor (Y).  
    - **Others**: If the image does not fit the above categories. for example, a schematic diagram or a map.

   Here, **X and Y** must be elements of the **material tetrahedron**, which includes four fundamental aspects:  
   - **Processing**: Refers to fabrication, synthesis, or treatment processes (e.g., heat treatment, annealing).  
   - **Structure**: Refers to the material’s internal or external structure (e.g., grain size, phase distribution).  
   - **Property**: Refers to physical, chemical, or mechanical properties (e.g., hardness, electrical conductivity).  
   - **Performance**: Refers to the overall effectiveness or functionality in an application (e.g., wear resistance, energy efficiency).  

3. **Identify the text block(s) that describe the image**:
   - Return a list of `block_index` values corresponding to the `"text"` blocks that explain or refer to the image's content, role, or observations.
   - These text blocks may include:
    - Paragraphs that refer to the figure (e.g., "as shown in Fig. 2...") 
    - Blocks that interpret, analyze, or explain the visual content.

---

### Expected Output

Your final output should be a **list of dictionaries**, where each dictionary corresponds to one image block. Each dictionary must contain:

```json
{
  "block_index": <int>,                    // index of the image block
  "microscopic_image": <true | false>,     // whether it's a microscopic image
  "function": <string>,                    // one of: "Characterizing X", "Revealing the influence from X to Y", or "Others"
  "described_by": [<int>, <int>, ...]      // block indices of the text blocks that describe the image
}
```

---

### Example

Given an image block at index 12 showing a microscopic view of grain boundaries, described in block 23:
```json
[
  {
    "block_index": 12,
    "microscopic_image": true,
    "function": "Characterizing Structure",
    "described_by": [23]
  }
]
```

If an image at block 25 is a diagram showing how heat treatment (Processing) affects grain structure (Structure) and results in higher hardness (Property), discussed in blocks 26–28:
```json
[
  {
    "block_index": 25,
    "microscopic_image": false,
    "function": "Revealing the influence from Processing to Property",
    "described_by": [26, 27, 28]
  }
]
```

### Final Notes

Output must be a clean **JSON array**, with each image as a separate object.