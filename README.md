# SHA-256 Truncation Collision Study

This repository hosts the code, data and results for:

**“Analysis of Truncated SHA-256 Collisions via GPU-Accelerated Monte Carlo for Quantum-Threat Contexts”**

---

## 📋 Overview

This project measures collision probabilities and performance of 32-bit and 64-bit truncated SHA-256 under both classical and near-term quantum-threat models. We use an NVIDIA A30 GPU with CUDA to run large-scale Monte Carlo simulations on up to 10 000 000 random 256-bit keys per batch.

Key deliverables:
- **Collision probability curves** (vs. batch size, vs. key length, heatmap)
- **Execution-time benchmarks** (throughput for millions of hashes)
- A clear **comparison with theoretical (birthday-paradox) predictions**
- Fully **reproducible code**, data and plots

---

## 🚀 Getting Started

### Prerequisites

- **CUDA Toolkit 12.2**  
- **NVIDIA A30** (or compatible GPU)  
- **Ubuntu 22.04 LTS** (or similar Linux)  
- **Python 3.10**  
- **PyCUDA 2022.1**, NumPy, Matplotlib, Pandas

### Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/cgkinyua/sha256-truncation-collision-study.git
   cd sha256-truncation-collision-study
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

pip install -r requirements.txt

.
├── data/           # (optional) small sample key files
├── src/            # CUDA kernels, Python simulation & plotting scripts
├── results/        # raw logs, PNG and PDF figures
├── README.md       # this file
├── requirements.txt
└── .gitignore

python src/monte_carlo_simulation.py \
  --truncation 32 \
  --key_length 256 \
  --batch_size 100000 \
  --trials 10
python src/plot_collision_probability_vs_num_keys.py \
  --truncation 32

Gitonga C.K. “Analysis of Truncated SHA-256 Collisions via GPU-Accelerated Monte Carlo for Quantum-Threat Contexts.” Journal Name, 2025.
Repository: https://github.com/cgkinyua/sha256-truncation-collision-study
