# LEPTODEEPVAX: In Silico Retrosynthesis and Strategic Disconnection of _Leptospira interrogans_ OMPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**LEPTODEEPVAX** is an end-to-end computational framework designed to automate the discovery of subunit vaccine candidates for _Leptospira interrogans_. By integrating **retrosynthetic logic** from organic chemistry with **Deep Learning (Transformer-CNN architectures)**, the system deconstructs complex outer membrane proteins (OMPs) into optimized, synthetically viable precursors known as synthons.

## 📋 Table of Contents

- [Core Architecture](#-core-architecture)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Mathematical Framework](#-mathematical-framework)
- [Citation](#-citation)

---

## 🔬 Core Architecture

The system utilizes a dual-network intelligence developed entirely from scratch to maintain complete independence from pre-trained biological libraries:

- **BioFeatureEngineer (`src/features.py`):** A custom biological encoder utilizing **Self-Attention** and **Positional Encoding** to capture long-range amino acid dependencies, coupled with a **Convolutional Block** (1D-CNN) to extract local structural motifs.
- **LeptoNetV2 (`src/architecture.py`):** A deep neural classifier designed to project 320-dimensional latent vectors into an immunogenic priority manifold using batch normalization and dropout for regularization.
- **Retrosynthetic Engine (`src/engine.py`):** Implements **Strategic Disconnection**, simulating the cleavage of peptide bonds to identify 15-mer synthons with optimal physicochemical reactivity scores.

---

## ⚙️ Methodology

1.  **Data Acquisition:** Automated programmatic retrieval of proteomic records via the **NCBI Protein Database** API.
2.  **De Novo Encoding:** Sequences are digitized into a 128-dimensional embedding space where the AI learns "biological grammar" through custom Transformer-based attention logic.
3.  **End-to-End Optimization:** The encoder and classifier are co-trained for 100 epochs using **Binary Cross-Entropy (BCE)** loss and the Adam optimizer.
4.  **Priority Ranking:** Targets are evaluated and ranked based on a heuristic proxy of charge density and flexibility.
5.  **Blueprint Generation:** The lead candidate is subjected to retrosynthetic deconstruction to generate a **Strategic Disconnection Map** for laboratory synthesis.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- pip (Python Package Installer)

### Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/LEPTODEEPVAX_PROJECT.git](https://github.com/yourusername/LEPTODEEPVAX_PROJECT.git)
   cd LEPTODEEPVAX_PROJECT
   ```

````

2. **Environment Initialization:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

````

3. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

---

## 💻 Usage

To execute the complete pipeline (Acquisition → Training → Disconnection → Visualization):

```bash
python main.py

```

### Module Breakdown

- **`src/acquisition.py`**: Interfaces with NCBI Entrez API for automated sequence mining.
- **`src/features.py`**: The "From Scratch" Transformer logic and CNN motif finders.
- **`src/engine.py`**: Executes the sliding-window disconnection algorithm to find the top 5 synthons.
- **`src/visualizer.py`**: Generates publication-quality figures, including PCA manifold clusters and synthon maps.

---

## 📁 Project Structure

```text
LEPTODEEPVAX_PROJECT/
├── data/                    # Scraped protein records (.csv)
├── models/                  # Saved .pth weights for Encoder and LeptoNet
├── plots/                   # Generated academic figures (Fig 1-4)
├── results/                 # Analysis logs and CSV output
├── src/                     # Source code modules
│   ├── acquisition.py       # NCBI Gateway
│   ├── architecture.py      # LeptoNetV2 Definition
│   ├── engine.py            # Disconnection Engine
│   ├── features.py          # Custom Attention Encoder
│   ├── preprocessing.py     # Data transformation
│   └── visualizer.py        # Academic Visualizer
├── main.py                  # Entry point for end-to-end pipeline
├── requirements.txt         # Frozen library dependencies
├── .gitignore               # Minimal ignore file (.venv, __pycache__)
└── README.md

```

---

## 📈 Mathematical Framework

The priority score () used for retrosynthetic labeling is derived from residues classified by density and flexibility. The label is calculated as:

Where Charge Residues and Flexible Residues .

---

## 📚 Citation

```bibtex
@inproceedings{amaya2025leptodeepvax,
  title={In Silico Retrosynthesis and Strategic Disconnection of Leptospira interrogans Outer Membrane Proteins for Subunit Vaccine Design},
  author={Amaya, Janssen Carl T. and Galagar, John Mark},
  booktitle={Proceedings of the CCS Research Conference},
  year={2025},
  organization={University of Cebu}
}

```

---

_Developed by Janssen Carl T. Amaya and John Mark Galagar | University of Cebu_
