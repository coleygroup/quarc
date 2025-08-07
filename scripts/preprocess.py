"""Preprocessing Pipeline for Condition Recommendation

This script implements a multi-stage preprocessing pipeline for reaction data:

1. Data Organization (--chunk-json)
    - Creates initial chunks from raw JSON files
    - Groups chunks for efficient parallel processing

2. Data Collection & Deduplication (--collect-dedup)
    - Extracts key information from reactions (SMILES, temperature, amounts)
    - Deduplicates reactions at condition-level
    - Uses parallel processing with local and global deduplication

3. Initial Filtering (--init-filter)
    - Filters reactions based on basic criteria:
     * Product/reactant/agent count limits
     * Maximum atom counts
     * RDKit parsability

4. Train/Val/Test Split (--split)
    - Splits data by document ID (75:5:20)
    - Ensures related reactions stay together

5. Stage-Specific Filtering (--stage{1,2,3,4}-filter)
    - Stage 1: Agent existence and amounts
    - Stage 2: Temperature existence and range
    - Stage 3: Reactant amount ratios and uniqueness
    - Stage 4: Agent amount ratios and solvent checks

6. All Filters (--all-filters)
    - Runs all filters to generate data with targets in all stages

Usage:
    python preprocess.py --config config.yaml [--chunk-json] [--collect-dedup]
                        [--generate-vocab] [--init-filter] [--split]
                        [--stage1-filter] [--stage2-filter] [--stage3-filter] [--stage4-filter]
                        [--all-filters] [--all]
"""

import argparse
import os
import yaml
from loguru import logger

from quarc.preprocessing.create_chunks import create_initial_chunks, regroup_chunks
from quarc.preprocessing.collect_deduplicate import collect_and_deduplicate_parallel
from quarc.preprocessing.final_deduplicate import merge_and_deduplicate
from quarc.preprocessing.generate_agent_class import generate_agent_class
from quarc.preprocessing.split_by_document import split_by_document
from quarc.preprocessing.initial_filter import run_initial_filters
from quarc.preprocessing.condition_filter import (
    run_stage1_filters,
    run_stage2_filters,
    run_stage3_filters,
    run_stage4_filters,
    run_all_filters,
)
from quarc.cli.quarc_parser import add_preprocess_opts
from quarc.settings import load as load_settings

cfg = load_settings()


def load_config(config_path) -> dict:
    """Loads preprocessing config and replace paths"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # replace placeholders with environment variables
    if "dirs" in config:
        dirs = config["dirs"]
        replacements = {
            "{raw_dir}": os.getenv("RAW_DIR", cfg.get("raw_data_path")),
            "{data_interim}": str(cfg.data_dir / "interim"),
            "{data_processed}": str(cfg.processed_data_dir),
            "{data_root}": str(cfg.data_dir),
            "{checkpoints}": str(cfg.checkpoints_dir),
        }
        for key, path in dirs.items():
            if isinstance(path, str):
                for template, replacement in replacements.items():
                    path = path.replace(template, replacement)
                dirs[key] = path

    return config


def setup_logging(config):
    """Setup logging for the entire preprocessing pipeline."""
    log_dir = cfg.logs_dir / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()

    logger.add(
        log_dir / "preprocessing.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {extra[stage]} | {message}",
        level="INFO",
    )

    if config.get("logging", {}).get("debug_logging", False):
        stages_with_debug = config["logging"].get("debug_stages", {})

        for stage_name, stage_enabled in stages_with_debug.items():
            if stage_enabled:
                logger.add(
                    log_dir / f"{stage_name}_debug.log",
                    filter=lambda record: record.extra.get("stage") == stage_name,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[stage]} | {message}",
                    level="DEBUG",
                )


def main():

    parser = argparse.ArgumentParser(description="preprocessing")
    add_preprocess_opts(parser)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)

    setup_logging(config)

    # Step 1: Data Organization
    if args.chunk_json or args.all:
        with logger.contextualize(stage="chunking"):
            create_initial_chunks(config)
            regroup_chunks(config)

    # Step 2: Data Collection & Deduplication
    if args.collect_dedup or args.all:
        with logger.contextualize(stage="data_collection"):
            collect_and_deduplicate_parallel(config)
            merge_and_deduplicate(config)

    # Step 2.5: Generate agent class
    if args.generate_vocab or args.all:
        with logger.contextualize(stage="generate_agent_class"):
            generate_agent_class(config)

    # Step 3: Initial Filtering
    if args.init_filter or args.all:
        with logger.contextualize(stage="initial_filter"):
            run_initial_filters(config)

    # Step 4: Train/Val/Test Split
    if args.split or args.all:
        with logger.contextualize(stage="split"):
            split_by_document(config)

    # Step 5: Stage-Specific Filtering
    if args.stage1_filter or args.all:
        with logger.contextualize(stage="stage1"):
            run_stage1_filters(config)

    if args.stage2_filter or args.all:
        with logger.contextualize(stage="stage2"):
            run_stage2_filters(config)

    if args.stage3_filter or args.all:
        with logger.contextualize(stage="stage3"):
            run_stage3_filters(config)

    if args.stage4_filter or args.all:
        with logger.contextualize(stage="stage4"):
            run_stage4_filters(config)

    if args.all_filters or args.all:
        with logger.contextualize(stage="all_filters"):
            run_all_filters(config)


if __name__ == "__main__":
    main()
