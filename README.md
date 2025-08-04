# QUARC (QUAtitative Recommendations of reaction Conditions)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)

QUARC is a data-driven model for recommending agents, temperature, and equivalence ratios for organic synthesis.

> [!IMPORTANT]
> The QUARC models used in the paper rely on the **[NameRxn](https://www.nextmovesoftware.com/namerxn.html) reaction classification codes** as part of the model input. Specifically, the reaction class is encoded as a one-hot vector, requiring access to the full NameRxn code mapping.
>
> Users with a Pistachio license can access the **2023Q4 version** of `Pistachio Reaction Types.csv` directly (2271 NameRxn classes + Unrecognized). Otherwise, they may contact the authors (xiaoqis@mit.edu) to obtain the specific files used in this work.
>
> For users wihtout NameRxn access, we are preparing an open-source version that removes this dependency. The back-end service is planned to be integrated into [ASKCOSv2](https://gitlab.com/mlpds_mit/askcosv2)'s july release ([release note](https://askcos-docs.mit.edu/release-notes/8-Changelogs.html)). In the meantime, an early-access implementation is also avaliable at the [quarc-oss](https://github.com/Xiaoqi-Sun/quarc-oss) repo.

## Quick Start (Inference Only)

If you just want to predict conditions for your reactions using the provided pretrained models:

### Step 1: Environment Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml -n quarc
conda activate quarc
pip install --no-deps -e .

# 2. Configure NameRxn Code Mapping (REQUIRED)
export PISTACHIO_NAMERXN_PATH="/path/to/your/Pistachio Reaction Types.csv"

# 3. Optional: Set data paths (uses defaults if not set)
export DATA_ROOT="~/quarc/data"
```

_The 2023Q4 version of  `Pistachio Reaction Types.csv` is required for compatibility with the pretrained models (requires 2272 classes for reaction class encoding). Using a different version may cause the model to fail._

### Step 2: Run Predictions

```bash
# Get predictions using the example input file
python scripts/inference.py \
    --input data/example_input.json \
    --output predictions.json \
    --config ffn_pipeline.yaml \
    --top-k 5
```

Results will be in `predictions.json` with recommended agents, temperatures, and amounts. Atom-mapped SMILES are required for the GNN models.

## Usage

### Inference with custom data

#### Input Format

```json
[
  {
    "rxn_smiles": "[CH3:1][O:2][C:3]...",
    "rxn_class": "1.8.7",
    "doc_id": "my_reaction_1"
  }
]
```

#### Model Options

```bash
# FFN models (works with any SMILES)
python scripts/inference.py \
    --input input.json \
    --output predictions.json \
    --config ffn_pipeline.yaml \
    --top-k 5

# GNN models (requires atom-mapped SMILES)
python scripts/inference.py \
    --input input.json \
    --output predictions.json \
    --config gnn_pipeline.yaml \
    --top-k 5
```

<!-- #### Output Format

Results include ranked predictions with confidence scores, agent SMILES, temperature ranges, and equivalence ratios. See [example output](data/example_output.json). -->

<!-- ### Benchmarking

To evaluate QUARC on your own test sets:

```bash
# Run evaluation with ground truth conditions
python scripts/evaluate.py \
    --input data/test_reactions.json \
    --config configs/eval_config.yaml
``` -->

## Development and Retraining

> **Note**: Requires Pistachio's density data and NameRxn access

### Environment Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml -n quarc
conda activate quarc
pip install --no-deps -e .

# 2. Configure paths
export DATA_ROOT="~/quarc/data"
export PISTACHIO_DENSITY_PATH="/path/to/density.tsv"
export PISTACHIO_NAMERXN_PATH="/path/to/Pistachio Reaction Types.csv"

# Alternatively, edit the config file
nano configs/quarc_config.yaml
```

### Data Preprocessing

Configure the preprocessing pipeline in `configs/preprocess_config.yaml`, replace the raw directory with your extracted Pistachio folder and other paths as needed.

```bash
python scripts/preprocess.py --config configs/preprocess_config.yaml --all
```

### Model Training

```bash
python scripts/ffn_train.py --stage 1 \
    --batch-size 256 \
    --max-epochs 30 \
    --save-dir ./train_log \
    --logger-name stage1 \
    --FP_radius 3 \
    --FP_length 2048 \
    --hidden_size 1024 \
    --n_blocks 2 \
    --init_lr 1e-4
```

or

```bash
python scripts/gnn_train.py --stage 1 \
    --batch-size 256 \
    --max-epochs 30 \
    --save-dir ./train_log \
    --logger-name stage1 \
    --graph_hidden_size 300 \
    --depth 3 \
    --hidden_size 1024 \
    --n_blocks 2 \
    --init_lr 1e-4
```

## References

If you find our code or model useful, we kindly ask that you consider citing our work in your papers.

```bibtex
@article{Sun2025quarc,
  title={Data-Driven Recommendation of Agents, Temperature, and Equivalence Ratios for Organic Synthesis},
  author={Sun, Xiaoqi and Liu, Jiannan and Mahjour, Babak and Jensen, Klavs F and Coley, Connor W},
  journal={ChemRxiv},
  doi={10.26434/chemrxiv-2025-4wzkh},
  year={2025}
}
```
