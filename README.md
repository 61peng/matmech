# **Material Mechanism Dataset Construction**

This repository provides the complete codebase and supporting resources for the paper
**_A Multimodal Dataset of Causal Mechanisms in Materials Science Literature_**,
including the full data processing pipeline, model training scripts, and usage documentation.

---

# **Repository Structure**

The repository is organized as follows:

```
.
├── instruction/          # Instruction templates and prompt data used for LLM-based generation
├── models/               # Model checkpoints, LoRA weights, and inference-ready model files
├── pipeline/             # End-to-end pipeline for tetrahedron extraction, image function parsing, and mechanism construction
├── output_file/          # Intermediate model outputs grouped by journal
│   ├── raw_data/         # Raw model generations stored in JSONL (one paper per line)
│   └── processed_data/   # Parsed JSON outputs derived from LLM responses
└── final_output/         # Final integrated dataset used for downstream annotation and analysis
```

Each directory contains additional documentation where necessary.

---

# **Reproduction**

## **Requirements**

```
pip install -r requirements.txt
```

## **Fine-tuning LLM with LoRA**

### **Dataset Construction**

Training and validation datasets are built under the `training_data/` directory.
Each training sample follows the format:

```json
{
  "doi": "xxx",
  "category": "tetrahedron extraction",
  "conversation": [
    {
      "human": "INSTRUCTION + PAPER CONTENT",
      "assistant": "{\"material object\": \"Silicon nanowires (SiNWs)\", \"Processing\": \"Bottom-imprint method using Al catalyst, UHV thermal evaporation, hot-embossing, and UHV CVD process\", \"Structure\": \"Vertically standing epitaxial Si nanowire arrays with controlled diameter and crystallographic orientation\", \"Properties\": \"Diameter control–structural property, crystallographic orientation–structural property\", \"Performance\": \"Compatibility with standard Si processing lines\", \"logical chain\": \"Processing → Structure → Performance\"}"
    }
  ]
}
```

Due to copyright restrictions, **training data are not distributed**.
Users must download the relevant papers and construct their own datasets following the above schema.

---

### **Fine-tuning Parameters and Hyperparameter Settings**

Fine-tuning configuration files are provided under:

```
config/tetrahedron_config.json
```

Key training parameters:

```
num_train_epochs: 10
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
max_seq_length: 13000
lr_scheduler_type: constant_with_warmup
gradient_checkpointing: false
optim: adamw_torch
fp16: true
```

Hyperparameters:

```
seed: 42
weight_decay: 0
max_grad_norm: 0.3
warmup_steps: 100
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 2e-4
```

Distributed training is performed using **DeepSpeed ZeRO Stage 3**, with configuration:

```
config/ds_z3_on_config.json
```

Experiments in the paper were conducted using **2 × NVIDIA H100 80GB GPUs**.
Please adjust batch size and parallelism settings according to your available hardware.

---

### **Start Training**

Run the training script:

```bash
deepspeed --num_gpus=2 --master_port 29600 train.py \
  --train_args_file config/training_config.json
```

After training, LoRA adapters are saved to the directory: `adapter/`

---

### **Launching the Model via vLLM**

```bash
vllm serve model/Llama-3.3-70B-Instruct \
  --max-model-len 131072 \
  --enable-lora \
  --lora-modules wgp=path/to/adapter \
  --port 8000 \
  --trust-remote-code \
  --tensor_parallel_size 2 \
  --max-lora-rank 64
```

This starts a fast inference API for downstream mechanism extraction.

## **Annotation Schemas**

This repository defines two core annotation schemas used in the construction of the **Material Mechanism Dataset**:
(1) the **Tetrahedron Recognition Schema**, which extracts high-level causal elements based on the MATERIALS SCIENCE TETRAHEDRON (Processing–Structure–Property–Performance), and
(2) the **Mechanism Analysis Schema**, which provides a fine-grained, multimodal representation of causal links supported by textual and visual evidence.

Both schemas are designed to ensure **structural consistency**, **machine readability**, and **high reproducibility** across the dataset.

---

### **1. Tetrahedron Recognition Schema**

The tetrahedron schema captures the essential causal elements of a materials science study and organizes them in a structured JSON format:

```json
{
  "material object": "<Name of the material>",
  "Processing": "<Description of the processing step or null>",
  "Structure": "<Description of the structure change or null>",
  "Properties": "<Property–Property Category or null>",
  "Performance": "<Performance aspect or null>",
  "logical chain": "<Processing → Structure → Property → Performance>"
}
```

**Field definitions:**

- **material object** — The material system being studied.
- **Processing** — Synthesis, fabrication, or treatment conditions.
- **Structure** — Microstructural or crystallographic features resulting from processing.
- **Properties** — Measured physical, chemical, or mechanical properties.
- **Performance** — Functional or application-level outcomes.
- **logical chain** — A symbolic representation of the causal pathway connecting tetrahedral elements.

This schema allows consistent extraction of high-level causal relationships from scientific text.

---

### **2. Mechanism Analysis Schema**

The mechanism analysis schema provides a **fine-grained, evidence-grounded representation** of each causal relationship identified in the paper. Each mechanism is expressed as an independent JSON object:

```json
[
  {
    "link": "Processing → Structure",
    "cause": "...",
    "effect": "...",
    "description": "One-sentence summary explaining how cause leads to effect",

    "experiment": {
      "name": "...",
      "type": "e.g., TEM, XRD, tensile test",
      "parameters": "...",
      "result": "...",
      "source": [block_index]
    },

    "images": [
      {
        "block_index": 14,
        "image description": "...",
        "microscopic": true,
        "source": [block_index]
      }
    ],

    "referenced_knowledge": [
      {
        "content": "...",
        "reference": ["[3] J. Mater. Sci. 2020, 55, 1234."],
        "source": [block_index]
      }
    ],

    "non-referenced_knowledge": [
      {
        "content": "...",
        "source": [block_index]
      }
    ],

    "mechanism": {
      "description": "Scientific explanation of the causal link",
      "source": [block_index],
      "reasoning_chain": [
        {
          "statement": "...",
          "type": "experimental result | image description | non-referenced knowledge | referenced knowledge | deductive reasoning | inductive reasoning"
        }
      ]
    }
  }
]
```

**Core components:**

- **link** — The causal relation category (e.g., Processing → Structure, Structure → Property).
- **cause / effect** — Text spans extracted directly from the paper.
- **description** — A concise natural-language summary of the causal relation.
- **experiment** — The experimental setup and result that support the mechanism, referenced by in-paper block indices.
- **images** — Microscopy or figure-based evidence tied to figure blocks.
- **referenced_knowledge** — External knowledge cited in the paper via references.
- **non-referneced_knowledge** — External knowledge not cited in the paper but reflects well-established domain principles.
- **mechanism** — The full mechanistic explanation, including a structured reasoning chain with explicit knowledge types.

This schema supports multimodal reasoning and enables downstream tasks such as mechanism verification, causal chain reconstruction, and visual–textual grounding analysis.

---

# **Usage**

## **Download the Dataset**

The complete dataset is publicly available on Figshare:

**[https://figshare.com/s/eab319e9f9744e789f25](https://figshare.com/s/eab319e9f9744e789f25)**

## **Basic Usage Examples**

### **Explore sample cases**

```bash
ls case_studies/
```

### **Load a JSON case and visualize figures**

Each sample contains:

- parsed text evidence
- segmented figure regions
- mechanism annotations

### **Download and extract the full dataset**

```bash
unzip matmech.zip -d matmech
```

Each DOI folder contains the complete multimodal content for one paper (text, images, metadata, mechanisms).

---

# **License**

<!-- **MIT License** -->

**Apache-2.0 License**

---

<!-- # **Contact / Maintainers**

For questions or issues:

- Maintainer: _[Your Name or Team]_
- Email: _[[maintainer@email.com](mailto:maintainer@email.com)]_
- Issue Tracker: Submit issues via the repository’s GitHub “Issues” section -->
