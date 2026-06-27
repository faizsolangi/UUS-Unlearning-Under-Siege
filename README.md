# Unlearning Under Siege: Robustness Risks of Certified Machine Unlearning

Official implementation for the paper:

**"Unlearning Under Siege: Robustness Risks of Certified Machine Unlearning"**
Faiz Ahmad
*Neurocomputing*, 2026.

## Overview

This repository contains the experiment code for studying how Newton-step certified machine unlearning affects adversarial robustness. The code reproduces all tables and figures in the paper.

**Key findings:**
- Certified removal noise at epsilon_priv=5.0 reduces adversarial accuracy by 66-69 percentage points (MNIST, Fashion-MNIST)
- For a 2-layer MLP, adversarial deletion orderings reduce PGD accuracy by 9.7pp at k=100
- Margin degradation is inherent to data removal, confirmed by retrain-from-scratch baseline

## Repository Structure

```
UUS/
├── UUS_Experiments.py      # Main experiments (logistic regression, 3 datasets)
├── UUS_Supplementary.py    # MLP experiments + retrain baseline
├── prepare_data.py         # Dataset preparation for Kaggle
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Experiments

| Experiment | Script | Output |
|---|---|---|
| 1. Deletion ordering effects | UUS_Experiments.py | Table 1, Figure 1 |
| 2. Margin degradation | UUS_Experiments.py | Table 2 |
| 3. Noise-robustness tradeoff | UUS_Experiments.py | Table 3, Figure 2 |
| 4. Targeted attack | UUS_Experiments.py | Table 4, Figure 3 |
| 5. MLP extension | UUS_Supplementary.py | Table 5, Figure 4-5 |
| 6. Retrain baseline | UUS_Supplementary.py | Table 6, Figure 6 |

## Quick Start

### Option A: Run on Kaggle (recommended, no GPU needed)

1. Create a new Kaggle Notebook (CPU only)
2. Add these datasets via "Add data":
   - `mnist-in-csv` by oddrationale
   - `fashionmnist` by zalando-research
   - `cifar10-python-in-csv` by fedesoriano
3. Upload the scripts and run:

```python
!python UUS_Experiments.py
!python UUS_Supplementary.py
```

### Option B: Run locally

```bash
pip install -r requirements.txt

# Place dataset CSVs in ./data/ directory, then:
python UUS_Experiments.py
python UUS_Supplementary.py
```

## Datasets

All datasets are publicly available:
- **MNIST** (digits 3 vs 8): [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Fashion-MNIST** (T-shirt vs Trouser): [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- **CIFAR-10** (airplane vs automobile): [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv)

## Configuration

Key parameters (set at the top of each script):

| Parameter | Value | Description |
|---|---|---|
| LAMBDA | 0.001 | L2 regularization strength |
| N_SAMPLES | 2000 | Samples per class |
| PGD_STEPS | 40 | PGD attack iterations |
| PGD_STEP_SIZE | 0.005 | PGD step size |
| DELETION_STEPS | [5,10,25,50,100,150,200] | Number of deletions to test |

## Runtime

| Script | Estimated Time (CPU) |
|---|---|
| UUS_Experiments.py | ~2-3 hours |
| UUS_Supplementary.py | ~2-3 hours |

All experiments run on CPU only (Kaggle free tier compatible).

## Citation

```bibtex
@article{ahmad2026unlearning,
  title={Unlearning Under Siege: Robustness Risks of Certified Machine Unlearning},
  author={Ahmad, Faiz},
  journal={Neurocomputing},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
