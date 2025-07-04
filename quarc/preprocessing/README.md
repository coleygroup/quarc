# Condition Recommendation Preprocessing

Data preprocessing pipeline for extracting and filtering reaction conditions from Pistachio database.

## Quick Start

<div style="font-size: 0.9em; overflow-y: auto;">

```bash
# Download raw data (if needed)
wget <url_to_be_provided> -O pistachio_23Q2.zip
unzip pistachio_23Q2.zip -d /path/to/extract/folder

# Install dependencies
python3 -m venv env
source env/bin/activate or .\env\Scripts\activate
pip install preprocessing

# Create yaml file
make_yaml --data_directory {directory_with_reaction_data_json_files} --output_directory {output_dir}

# Run complete pipeline
preprocess --config {output_dir}/preprocess_config.yaml --all

# Or run individual steps as needed:

# Data organization
preprocess --config {output_dir}/preprocess_config.yaml --chunk_json

# Data collection
preprocess --config {output_dir}/preprocess_config.yaml --collect_dedup

# Agent vocabulary generation
preprocess --config {output_dir}/preprocess_config.yaml --generate_vocab

# Initial filtering
preprocess --config {output_dir}/preprocess_config.yaml --init_filter

# Train/val/test split
preprocess --config {output_dir}/preprocess_config.yaml --split

# Stage 1 filtering
preprocess --config {output_dir}/preprocess_config.yaml --stage1_filter

# Stage 2 filtering
preprocess --config {output_dir}/preprocess_config.yaml --stage2_filter

# Stage 3 filtering
preprocess --config {output_dir}/preprocess_config.yaml --stage3_filter

# Stage 4 filtering
preprocess --config {output_dir}/preprocess_config.yaml --stage4_filter
```

</div>

## Pipeline Overview

1. **Data Organization**: Creates manageable chunks from raw Pistachio JSON files
2. **Data Collection**: Extracts reaction conditions (temperature, amounts) and deduplicates
3. **Agent Vocabulary Generation**: Generates standardized agent vocabulary used by `agent_encoder` and `agent_standardizer`.
4. **Initial Filtering**: Basic reaction validation and filtering
5. **Train/Val/Test Split**: Document split (75:5:20)
6. **Stage-Specific Filtering**: Creates filtered datasets for each stage:
   - Stage 1: Agent prediction
   - Stage 2: Temperature prediction
   - Stage 3: Reactant amount prediction
   - Stage 4: Agent amount prediction

See [documentation.md](./docs/documentation.md) for:

- Detailed pipeline explanation
- Data structure specifications
- Filtering criteria
- Implementation details

## Configuration

Key parameters in `{output_dir}/preprocess_config.yaml`:

Make sure you properly update the data directories properly.

```yaml
dirs:
  raw_dir: "/path/to/pistachio/extract" # Raw data location

chunking:
  num_workers: 8 # Number of parallel workers
  chunk_size: 500000 # Reactions per chunk

generate_agent_class:
  minocc: 50 # Minimum occurrences for standard agents
  metal_minocc: 50 # Minimum occurrences for metal-containing agents

initial_filter:
  length_filters: # Molecule count limits
    product:
      min: 1
      max: 1 # Single product only
    reactant:
      min: 1
      max: 5 # 1-5 reactants
    agent:
      min: 0
      max: 5 # 0-5 agents
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

## Output Structure

```
data/
├── interim/                    # Intermediate files
│   ├── 23Q2_dump/             # Initial chunks
│   ├── 23Q2_grouped/          # Regrouped chunks
│   └── split/                 # Train/val/test splits
└── processed/                 # Final filtered datasets
    ├── agent_encoder/        # Agent vocabulary files
    │   ├── agent_encoder_list.json
    │   ├── agent_other_dict.json
    │   └── agent_rules_v1.json
    ├── stage1/               # Agent prediction
    ├── stage2/               # Temperature prediction
    ├── stage3/               # Reactant amount prediction
    └── stage4/               # Agent amount prediction
```

## Citation

If you use this code, please cite:

```bibtex
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```
