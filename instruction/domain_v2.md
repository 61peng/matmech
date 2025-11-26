Given structured data extracted from a materials science research paper. The data contains two fields:

1. "tetrahedron": a summary of the material under study, including its composition ("material object") and its relationships among "processing", "structure", "properties", and "performance" — following the Materials Tetrahedron framework.

2. "mechanism": a list of detailed cause-effect relationships that describe how changes in processing lead to changes in structure, how structure affects properties, and how properties impact performance.

Based on this information, determine the **material types** the material belongs to.

You should assign **one or more** of the following **material types**, based primarily on **composition**, **microstructure**, and **characteristic scale** (not application).

### Your classification should be selected from this controlled list:

- **Crystalline Material** – if the material exhibits a long-range ordered structure (e.g., single crystal, polycrystalline phase, epitaxial film).
- **Nanomaterial** – if the material has nanoscale features (e.g., quantum dots, nanoparticles, nanowires, nanosheets).
- **Composite Material** – if the material combines two or more distinct phases (e.g., polymer–ceramic composite, fiber-reinforced metal).
- **Polymer** – if the material is made of macromolecular organic chains (e.g., PI, PVDF, PEEK).
- **Ceramic** – if the material is inorganic and non-metallic, usually brittle and processed at high temperatures (e.g., SiO₂, Al₂O₃, BaTiO₃).
- **Metals and Alloys** – if the material is metallic or a combination of metals (e.g., Cu, NiTi alloy, high-entropy alloy).
- **Biomaterial** – if the material is designed for or inherently suitable for interaction with biological systems (e.g., hydroxyapatite, biodegradable scaffolds).
- **Coatings and Thin Films** – if the material is deposited as a thin layer, regardless of composition (e.g., TiN coating, ZnO film).
- **Paper and Wood-based Materials** – if the material is based on cellulose, wood, or derived biomass (e.g., nanocellulose film, biochar–wood composite).

> A material can belong to **multiple categories** if justified by structure or composition (e.g., a crystalline nanoparticle coating can be both _Crystalline_, _Nanomaterial_, and _Coatings and Thin Films_).

---

### Output format:

```json
["Material Type 1", "Material Type 2", ...]
```

---

### Notes for classification:

- Use **composition** (e.g., polymer, oxide, metal) as the **primary basis**.
- Use **morphology/scale** (e.g., nanoparticle, thin film, single crystal) as a **secondary basis**.
- Ignore purely functional or application-specific labels (e.g., “sensor material” is not a valid category).
- Avoid "Functional Material" or "Inorganic Material" as generic labels unless you have no better structural class.
- Output a list of mateirial Types (e.g., `["Nanomaterial", "Ceramic"]`). Do not output any explanation or additional text.
