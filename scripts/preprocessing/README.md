# Preprocessing condition data for QUARC

Data preprocessing pipeline for extracting and filtering reaction conditions from Pistachio database.

## Quick Start

### Prerequisites

- Download and unzip Pistachio database JSON files
- Ensure the QUARC conda environment is activated (see main repository README for installation)

### Usage

```bash
# 1. Create preprocessing configuration
python make_preprocessing_yaml.py \
    --data_directory /path/to/pistachio/json/files \
    --output_directory /path/to/your/output \
    --chunk_size 50000

# 2. Run complete pipeline
preprocess --config /path/to/your/output/preprocessing.yaml --all

# Or run individual steps as needed:

# Data organization
preprocess --config /path/to/your/output/preprocessing.yaml --chunk_json

# Data collection and deduplication
preprocess --config /path/to/your/output/preprocessing.yaml --collect_dedup

# Agent vocabulary generation
preprocess --config /path/to/your/output/preprocessing.yaml --generate_vocab

# Initial filtering
preprocess --config /path/to/your/output/preprocessing.yaml --init_filter

# Train/val/test split
preprocess --config /path/to/your/output/preprocessing.yaml --split

# Stage-specific filtering
preprocess --config /path/to/your/output/preprocessing.yaml --stage1_filter
preprocess --config /path/to/your/output/preprocessing.yaml --stage2_filter
preprocess --config /path/to/your/output/preprocessing.yaml --stage3_filter
preprocess --config /path/to/your/output/preprocessing.yaml --stage4_filter
```

## Pipeline Overview

1. **Data Organization**: Creates manageable chunks from raw Pistachio JSON files
2. **Data Collection**: Extracts reaction conditions (temperature, amounts) and deduplicates
3. **Agent Vocabulary Generation**: Generates standardized agent vocabulary used by `agent_encoder` and `agent_standardizer`
4. **Initial Filtering**: Basic reaction validation and filtering
5. **Train/Val/Test Split**: Document split (75:5:20)
6. **Stage-Specific Filtering**: Creates filtered datasets for each stage:
   - Stage 1: Agent prediction
   - Stage 2: Temperature prediction
   - Stage 3: Reactant amount prediction
   - Stage 4: Agent amount prediction

See [documentation.md](../../quarc/preprocessing/documentation.md) for:

- Detailed pipeline explanation
- Data structure specifications
- Filtering criteria
- Implementation details

## Configuration

The `make_preprocessing_yaml.py` script creates a configuration file with your specified paths. Key configurable parameters include:

```yaml
dirs:
  raw_dir: "/your/path/to/pistachio/json/files" # Raw data location
  # All other paths are automatically generated relative to output_directory

chunking:
  num_workers: 8 # Number of parallel workers
  chunk_size: 500000 # Reactions per chunk

generate_agent_class:
  minocc: 50 # Minimum occurrences for standard agents
  metal_minocc: 50 # Minimum occurrences for metal-containing agents

initial_filter:
  length_filters: # Molecule count limits
    product: # Single product only
      min: 1
      max: 1
    reactant: # 1-5 reactants
      min: 1
      max: 5
    agent:  0-5 agents
      min: 0
      max: 5 #
  atom_filters: # Size limits
    max_reactant_atoms: 50
    max_product_atoms: 50

stage_2_filter:
  temperature_range: # Temperature limits (Celsius)
    lower: -100
    upper: 200

stage_3_filter:
  ratio_range: # Reactant molar ratio limits
    lower: 0.001
    upper: 7.0

stage_4_filter:
  ratio_range: # Agent molar ratio limits
    lower: 0.001
    upper: 1000.0 # Higher limit for solvents
```

For manual configuration editing, see `preprocess_config.yaml` as a template.

## Output Structure

```

quarc/
├── logs/
│   └── preprocessing/       # Processing logs
└── data/
    ├── interim/             # Intermediate files
    │   ├── 23Q2_dump/       # Initial chunks
    │   ├── 23Q2_grouped/    # Regrouped chunks
    │   └── split/           # Train/val/test splits
    └── processed/           # Final filtered datasets
        ├── agent_encoder/   # Agent vocabulary files
        │   ├── agent_encoder_list.json
        │   ├── agent_other_dict.json
        │   └── agent_rules_v1.json
        ├── stage1/         # Agent prediction data
        ├── stage2/         # Temperature prediction data
        ├── stage3/         # Reactant amount prediction data
        └── stage4/         # Agent amount prediction data
```

## Citation

If you use this code, please cite:

```bibtex
@article{Sun2024quarc,
  title={Data-Driven Recommendation of Agents, Temperature, and Equivalence Ratios for Organic Synthesis},
  author={Sun, Xiaoqi and Liu, Jiannan and Mahjour, Babak and Jensen, Klavs F and Coley, Connor W},
  year={2025}
}
```
