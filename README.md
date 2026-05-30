# Triadic System

**Triadic System** is the reference implementation of the  
**Dynamic Modular Language Graph (DMLG)** — a minimal, transparent,  
fully functional language‑generation engine.

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

### **1. `dmlg/` — The Core Engine **  
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
- **ModeratedAgent** — agent that is moderated by ensemble of other agents

These classes form the complete DMLG engine.

---

### **2. `engine/` — System Helpers (runners live in the project root)**

The `engine/` package contains the helper classes used by the runnable
entry points (“mains”).  

It does **not** contain the runners themselves — all seven runners are
located in the **project root**.

`engine/` provides the orchestration layer around the core DMLG engine:

- **TriadicTrainer** — curriculum loading, exploration, training  
- **TriadicWriter** — story generation utilities  
- **TriadicSystem** — environment setup for the interactive demo  

---

## Runners

There are **eight official runners**, each demonstrating a different  
aspect of the system.

### **1. Interactive Demo**  
Run the full multi‑agent DMLG system in a prompt‑driven loop.

- Regex‑based prompt parsing  
- Model selection (`aesop`, `forest`, `hyde`, `observatory`, `poet`, `mix`, `distill`, `multi`)  
- Beam search toggle  
- Repeat last prompt  
- Quit commands  
- Story generation via:  
  `15 lines and model forest`

---

### **2. Training Runner**  
Train a DMLG agent from curriculum data.

- Exploration epochs (transition map + page creation)  
- Training epochs (neural page networks)  
- Saves model under `triadic-data/toy-system/<model>/`

---

### **3. Generation Runner**  
Generate stories using:

- prompts  
- keyword‑driven sequence embeddings  
- optional beam search  
- grammar + semantic validation  

---

### **4. Ensemble Writer Runner**  
Generate stories using a heterogeneous ensemble of agents trained with  
different activation functions.

Demonstrates:

- stylistic diversity  
- activation‑driven specialization  
- ensemble sampling  

---

### **5. Single‑Agent Distillation Runner**  
Generate purified canonical datasets from a single agent.

Useful for:

- compression  
- cleanup  
- re‑training  

---

### **6. Activation Visualization Runner**  
Plot all learned activation functions inside a trained model.

Shows:

- page‑wise learned non‑linearities  
- spikes, gates, hybrid SiLU/GELU shapes  
- emergent activation behaviour  

---

### **7. Adaptive Activation Function Demo**  
Demonstrates how a small trainable activation‑MLP can learn complex  
non‑linear functions and outperform static activations inside a larger model.

---

### **8. Moderated Agent Runner **
Demonstrates how DMLG agent can be moderated by an ensemble of other DMLG agents.
Also shows how the output quality of weaker models can be elevated by moderation.
This illustrates the usage of DMLG agents for language model moderation.

---

## Purpose

The Triadic System serves as:

- **a reference implementation** of the DMLG architecture  
- **a research platform** for structural language models  
- **a demonstration** of minimal, interpretable generative systems  
- **a toolkit** for training, distillation, ensembles and activation analysis  
- **a companion** to *The Triadic Cosmos* DMLG book and paper

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

## Repository Structure

The repository contains:

- `/` — the runners  
- `dmlg/` — the full DMLG engine  
- `engine/` — helper classes  
- `README.md` — this document  

All runners can be executed independently.

---

## Data Location (triadic-data)

This project does **not** store its own data.  
All datasets, trained models, and generated outputs are stored in the  
**`triadic-data`** repository, inside:
triadic-data/toy-system/


This mirrors the structure used by all other Triadic Cosmos toy projects.

---

## Data stored in *triadic-data/toy-system/*

### **1. Curriculum texts**
- Original texts authored for the Triadic Cosmos  
- Public‑domain texts (e.g., Project Gutenberg) with their respective licenses  
- Canonicalised versions (lemma-normalised, structurally reduced, grammar-tokenised)

### **2. Trained models**
- Binary `.bin` model files generated by the DMLG training pipeline  
- One model per agent  
- Fully reproducible from curriculum + configuration

### **3. Example outputs**
- Generated stories  
- Distilled curricula  
- Ensemble outputs  
- Beam-search samples  
- Activation‑visualisation data (optional)

All data in *triadic-data* is licensed under **MIT**,  
which allows reuse and experimentation.

---

## Licensing Clarification

Although the **data** in *triadic-data* is MIT‑licensed,  
the **triadic-system codebase** is licensed under **AGPL‑3.0**.

This means:

- MIT‑licensed data **does not** weaken, override, or bypass  
  the AGPL‑3.0 obligations of this repository.  
- Using the data is free and unrestricted,  
  but any modifications or extensions of the **code** must comply with AGPL‑3.0.  
- The MIT data is provided for transparency, reproducibility, and research —  
  **not** as a loophole for relicensing or circumventing AGPL‑3.0.

In short:

> **Data is MIT. Code is AGPL‑3.0.  
> MIT data does not change the license of the code.**

---

## Relationship Between Repositories

The Triadic Cosmos ecosystem uses a clean separation:

- **triadic-system** → the runnable DMLG engine (AGPL‑3.0)  
- **triadic-data** → all datasets, models, and outputs (MIT)  
- **triadic-cosmos** → the theoretical foundations (CC BY‑NC‑ND 4.0)  
- **triadic-toys** → Java toy universes (Apache 2.0)  

This separation ensures:

- legal clarity  
- reproducibility  
- modularity  
- clean licensing boundaries  

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
