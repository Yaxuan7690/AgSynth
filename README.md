# AgSynth: A Multi-Agent Reverse Synthesis Framework for Multimodal Mathematical Reasoning Data Generation

A **reverse synthesis** framework that decomposes seed math problems into logical cores and reconstructs them with *structurally novel* visual forms using a **5-LLM ReAct (Reasoning + Acting) 4+1 pipeline**, ensuring high-fidelity, diverse, and rigorously validated multimodal reasoning data.

---

## ✨ Core Innovations

- **Reverse Synthesis Paradigm**: Logic-first decomposition → visual-form reconstruction (vs. superficial augmentation).
- **ReAct 4+1 Architecture**:  
  `ConceptDesigner` → `VisualSpecifier` → `Painter` → `QA Generator` + `Quality Rater`  
  Each agent gets ≤3 retries with memory & reflection.
- **Strict Visual Innovation**: Forces *structural* changes (e.g., balance scale → gear train), not color/number swaps.
- **Hybrid QA**: Rule-based checks (40%) + LLM scoring (60%) → auto-reject if score < 6.

---

## 🏗️ Agent Roles & Outputs

| Agent             | Role                  | Key Output                     |
|-------------------|-----------------------|--------------------------------|
| `ConceptDesigner` | Logic Core Extractor  | `core_logic_kernel`, `concept_design` |
| `VisualSpecifier` | Visual Innovator      | `visual_form_choice`, `detailed_image_spec` |
| `Painter`         | Renderer              | Safe `matplotlib` code        |
| `QA Generator`    | Text Synthesizer      | Question + Answer + Steps     |
| `Quality Rater`   | Scorer & Reviewer     | Score (1–10) + feedback       |

