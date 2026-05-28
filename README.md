# Triadic System

**Triadic System** is the reference implementation of the  
**Dynamic Modular Language Graph (DMLG)** — a minimal, transparent,  
fully functional language‑generation engine built from only sixteen  
Python classes.

Where modern language models rely on scale, the DMLG relies on  
**structure**: deterministic tokens, canonical grammar, fixed‑position  
context windows, page‑wise neural micro‑models, and rule‑based  
syntactic constraints.  

The result is the smallest complete language‑generation system ever  
constructed, capable of producing coherent multi‑sentence stories  
using CPU‑only PyTorch.

All components are implemented in Python and can be run independently.

---

## Included Components

The project is organized into two packages:

### **1. `dmlg/` — The Core Engine (16 classes)**  
This package contains the full DMLG architecture:

- **Configuration** — global model, context, curriculum and generation settings  
- **WriterEnvironment** — binds configuration, grammar and semantics  
- **GrammarEngine** — natural ↔ canonical grammar engine  
- **SemanticEngine** — semantic sanity checks  
- **Token**, **TokenDictionary**, **TokenPage**, **TokenMapping** — deterministic tokens and pages  
- **SentenceEncoder**, **EncodedSentence** — compressed sentence representations  
- **ContextWindow**, **ModelInput**, **InputEncoder** — deterministic context and model input  
- **TransitionMap** — learned graph of allowed token transitions  
- **NeuralNetwork** — tiny 3‑layer MLP with optional activation‑MLP  
- **PagedNetwork** — page‑wise prediction networks  
- **RuleBasedFilter** — syntactic firewall  
- **Curriculum**, **CurriculumStory**, **CurriculumSentence** — canonical training data  
- **WriterSentence**, **WriterStory** — natural output representation  
- **WriterAgent** — a micro‑expert trained on a curriculum  
- **MultiAgent** — ensemble of micro‑experts  

These sixteen classes form the complete DMLG engine.

---

### **2. `engine/` — Runners and System Helpers**

This package contains the runnable entry points (“mains”) and helper  
classes for training, generation, distillation, ensembles and demos.

There are **seven official runners**, each demonstrating a different  
aspect of the system.

---

## Runners

### **1. Interactive Demo**  
Run the full multi‑agent DMLG system in a prompt‑driven loop.

- Regex‑based prompt parsing  
- Model selection (`aesop`, `forest`, `hyde`, `observatory`, `poet`, `mix`, `distill`, `multi`)  
- Beam search toggle  
- Repeat last prompt  
- Quit commands  
- Story generation via:  
  `15 lines and model forest`

### **2. Training Runner**  
Train a DMLG agent from curriculum data.

- Exploration epochs (transition map + page creation)  
- Training epochs (neural page networks)  
- Saves model under `triadic-data/toy-system/<model>/`  

### **3. Generation Runner**  
Generate stories using:

- prompts  
- keyword‑driven sequence embeddings  
- optional beam search  
- grammar + semantic validation  

### **4. Ensemble Writer Runner**  
Generate stories using a heterogeneous ensemble of agents trained with  
different activation functions.

Demonstrates:

- stylistic diversity  
- activation‑driven specialization  
- ensemble sampling  

### **5. Ensemble Distillation Runner**  
Generate large synthetic curricula from an ensemble.

Used for:

- dataset expansion  
- style blending  
- curriculum distillation  

### **6. Single‑Agent Distillation Runner**  
Generate purified canonical datasets from a single agent.

Useful for:

- compression  
- cleanup  
- re‑training  

### **7. Activation Visualization Runner**  
Plot all learned activation functions inside a trained model.

Shows:

- page‑wise learned non‑linearities  
- spikes, gates, hybrid SiLU/GELU shapes  
- emergent activation behaviour  

---

## Purpose

The Triadic System serves as:

- **a reference implementation** of the DMLG architecture  
- **a research platform** for structural language models  
- **a demonstration** of minimal, interpretable generative systems  
- **a toolkit** for training, distillation, ensembles and activation analysis  
- **a companion** to *The Triadic Cosmos* book and papers  

It complements the Triadic Cosmos ecosystem by providing runnable,  
transparent examples of the concepts described there.

---

## Licensing

- **Source code** in this repository is released under the  
  **AGPL‑3.0 License**.  
  This ensures openness, reproducibility and shared improvements.

- **Books and papers** of the Triadic Cosmos ecosystem are licensed  
  separately under **CC BY‑NC‑ND 4.0**.  
  They may not be reused, modified or incorporated into derivative works.

The DMLG engine is **not** a loophole for reusing protected textual  
content. It is a standalone computational demonstration.

---

## Structure

The repository contains:

- `dmlg/` — the 16‑class DMLG engine  
- `engine/` — runners, helpers and demos  
- `triadic-data/` — models, curricula and outputs  
- `README.md` — this document  

All runners can be executed independently.

---

## Requirements

- Python 3.10+  
- CPU‑only PyTorch  
- spaCy + pyinflect + inflect  
- language‑tool‑python  
- NumPy, matplotlib  

No GPU is required.

---

## Status

The Triadic System is stable, runnable and designed for educational  
clarity. It forms the computational backbone of the DMLG reference  
implementation and the Triadic Cosmos Library.
