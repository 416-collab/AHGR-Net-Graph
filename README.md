# AHGR-Net: Adaptive Heterogeneous Graph Reasoning for Interpretable Multimodal Sentiment Analysis

[![---)]()
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## Overview

AHGR-Net is a unified framework for interpretable multimodal sentiment analysis that integrates three components:

- **Similarity-guided context mining** — retrieves semantically relevant preceding utterances to capture discourse-level sentiment dependencies
- **Aspect-aware heterogeneous graph construction** — connects target utterances, contextual utterances, and semantic aspect nodes through five typed relation categories
- **Relation-aware graph attention** — propagates information across typed relations to produce interpretable, aspect-grounded sentiment representations

The framework is evaluated on CMU-MOSI, CMU-MOSEI, and CH-SIMS and demonstrates consistent improvements across regression and classification metrics.

---

## Repository Structure

```
GRAPH_SA_DC2M/
├── configs/                        # YAML configuration files
│   ├── config_mosi.yaml
│   ├── config_mosei.yaml
│   └── config_mosi_trimodal.yaml
│
├── data/                           # Dataset loaders
│   ├── __init__.py
│   ├── mosi_pkl_dataset.py
│   ├── mosei_sequence_dataset.py
│   └── mosei_features_dataset.py
│
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── dc2m_baseline.py            # DC2M context-aware baseline
│   └── graph_sa.py                 # AHGR-Net graph reasoning model
│
├── scripts/                        # Training and evaluation scripts
│   ├── train_mosi.py
│   ├── train_mosei.py
│   ├── train_chsims_ahgr_direct.py
│   ├── chsims_final_ensemble.py
│   ├── train_graph_sa.py
│   ├── train_graph_sa_ablation.py
│   ├── ablate_mosi_graph_sa.py
│   └── download_cmu.py
│
├── outputs/                        # Final result tables
│   ├── AHGR_Net_MOSI_results.csv
│   ├── AHGR_Net_MOSEI_results.csv
│   ├── AHGR_Net_CHSIMS_ensemble_results.csv
│   ├── AHGR_Net_CHSIMS_seed_results.csv
│   └── AHGR_Net_MOSI_ablation_results.csv
│
└── README.md
```

---

## Datasets

Three public multimodal sentiment datasets are used. Raw data is **not included** in this repository due to size and licensing. Download and place under `data/raw/`:

```
data/raw/
├── MOSI/
├── MOSEI/
└── CHSIMS/
```

| Dataset | Language | Utterances | Label Range |
|---|---|---|---|
| CMU-MOSI | English | 2,199 | [-3, 3] |
| CMU-MOSEI | English | 22,852 | [-3, 3] |
| CH-SIMS | Chinese | 2,386 | [-1, 1] |

**Download links:**
- [CMU-MOSI](https://www.kaggle.com/datasets/reganwillis/cmu-mosi)
- [CMU-MOSEI](https://www.kaggle.com/datasets/gnurtqh/cmu-mosei)
- [CH-SIMS](https://thuiar.github.io/sims.github.io/chsims)

---

## Installation

```bash
git clone https://github.com/mshaban7416/Graph_AHGR_NET.git
cd Graph_AHGR_NET
pip install -r requirements.txt
```

---

## Training

### CMU-MOSI
```bash
python scripts/train_mosi.py --config configs/config_mosi.yaml
```

### CMU-MOSEI
```bash
python scripts/train_mosei.py --config configs/config_mosei.yaml
```

### CH-SIMS (Direct single model)
```bash
python scripts/train_chsims_ahgr_direct.py
```

### CH-SIMS (5-seed ensemble)
```bash
python scripts/chsims_final_ensemble.py
```

### Ablation Study (CMU-MOSI)
```bash
python scripts/ablate_mosi_graph_sa.py
```

---

## Evaluation

Result tables are available in `outputs/`:

| File | Description |
|---|---|
| `AHGR_Net_MOSI_results.csv` | CMU-MOSI main results |
| `AHGR_Net_MOSEI_results.csv` | CMU-MOSEI main results |
| `AHGR_Net_CHSIMS_ensemble_results.csv` | CH-SIMS 5-seed ensemble |
| `AHGR_Net_CHSIMS_seed_results.csv` | CH-SIMS per-seed results |
| `AHGR_Net_MOSI_ablation_results.csv` | CMU-MOSI ablation study |

---

## Key Implementation Details

- **Text encoder:** RoBERTa-base
- **Acoustic features:** COVAREP + Data2Vec-base
- **Visual features:** Facial action units
- **Hidden dimension:** 256
- **Graph layers:** 2 relation-aware GAT layers (4 heads → 1 head)
- **Context size K:** 3 for CMU-MOSI, 2 for CMU-MOSEI and CH-SIMS
- **Optimizer:** AdamW (lr=5e-6, weight decay=0.01)
- **Batch size:** 16
- **Max epochs:** 20 with early stopping (patience=5)

Full hyperparameter details are in the paper appendix (Table A1).

---

## Citation

If you use this code or the AHGR-Net framework in your research, please cite:

```bibtex
coming soon...
```

---



---

## Acknowledgments

The authors acknowledge the School of Software, Xi'an University of Technology, for academic guidance and research support. We also thank the developers of the CMU-MOSI, CMU-MOSEI, and CH-SIMS datasets.
