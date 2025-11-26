You are given structured data extracted from a materials science research paper. The data contains two fields:

1. "tetrahedron": a summary of the material under study, including its composition ("material object") and its relationships among "processing", "structure", "properties", and "performance" — following the Materials Tetrahedron framework.

2. "mechanism": a list of detailed cause-effect relationships that describe how changes in processing lead to changes in structure, how structure affects properties, and how properties impact performance.

Based on this information, determine the chemical elements involved in the target material object. Output a list of element symbols (e.g., `["C", "H", "O"]`). Do not output any explanation or additional text.
