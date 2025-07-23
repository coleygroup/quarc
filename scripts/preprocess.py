"""Preprocessing Pipeline for Condition Recommendation

This script implements a multi-stage preprocessing pipeline for reaction data:

1. Data Organization (--chunk_json)
    - Creates initial chunks from raw JSON files
    - Groups chunks for efficient parallel processing

2. Data Collection & Deduplication (--collect_dedup)
    - Extracts key information from reactions (SMILES, temperature, amounts)
    - Deduplicates reactions at condition-level
    - Uses parallel processing with local and global deduplication

3. Initial Filtering (--init_filter)
    - Filters reactions based on basic criteria:
     * Product/reactant/agent count limits
     * Maximum atom counts
     * RDKit parsability

4. Train/Val/Test Split (--split)
    - Splits data by document ID (75:5:20)
    - Ensures related reactions stay together

5. Stage-Specific Filtering (--stage1/2/3/4)
    - Stage 1: Agent existence and amounts
    - Stage 2: Temperature existence and range
    - Stage 3: Reactant amount ratios and uniqueness
    - Stage 4: Agent amount ratios and solvent checks

Usage:
    python preprocess.py --config config.yaml [--chunk_json] [--collect_dedup]
                        [--generate_vocab] [--init_filter] [--split]
                        [--stage1_filter] [--stage2_filter] [--stage3_filter] [--stage4_filter] [--all]
"""

import argparse
import yaml
from pathlib import Path
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
)


def load_config(config_path) -> dict:
    """Loads preprocessing configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """Setup logging for the entire preprocessing pipeline."""
    # Convert string to Path object
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(exist_ok=True)
    logger.remove()

    # Main pipeline logger - captures all INFO level messages
    logger.add(
        log_dir / "preprocessing.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {extra[stage]} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="a",
        serialize=False,
    )

    # Stage-specific debug loggers - only capture messages when within appropriate context
    if config["logging"]["debug_logging"]:
        stages_with_debug = config["logging"]["debug_stages"]

        for stage_name, stage_enabled in stages_with_debug.items():
            if stage_enabled:
                logger.add(
                    log_dir / f"{stage_name}_debug.log",
                    filter=lambda record: record.extra.get("stage") == stage_name,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[stage]}} | {message}",
                    level="DEBUG",
                    mode="a",
                    encoding="utf-8",
                    serialize=False,
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Run reaction preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--chunk_json",
        dest="chunk_json_enabled",
        action="store_true",
        help="Run data organization step",
    )
    parser.add_argument(
        "--collect_dedup",
        dest="collect_dedup_enabled",
        action="store_true",
        help="Run data collection and deduplication",
    )
    parser.add_argument(
        "--generate_vocab",
        dest="generate_vocab_enabled",
        action="store_true",
        help="Run agent class generation",
    )
    parser.add_argument(
        "--init_filter",
        dest="init_filter_enabled",
        action="store_true",
        help="Run initial filtering",
    )
    parser.add_argument(
        "--split", dest="split_enabled", action="store_true", help="Run train/val/test split"
    )
    parser.add_argument("--stage1_filter", action="store_true", help="Run agent filtering stage")
    parser.add_argument(
        "--stage2_filter", action="store_true", help="Run temperature filtering stage"
    )
    parser.add_argument(
        "--stage3_filter", action="store_true", help="Run agent amount filtering stage"
    )
    parser.add_argument(
        "--stage4_filter", action="store_true", help="Run reactant amount filtering stage"
    )
    parser.add_argument("--all", dest="run_all", action="store_true", help="Run complete pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    # Step 1: Data Organization
    if args.chunk_json_enabled or args.run_all:
        with logger.contextualize(stage="chunking"):
            create_initial_chunks(config)
            regroup_chunks(config)
    # Step 2: Data Collection & Deduplication
    if args.collect_dedup_enabled or args.run_all:
        with logger.contextualize(stage="data_collection"):
            collect_and_deduplicate_parallel(config)  # collect + local deduplication
            merge_and_deduplicate(config)  # final deduplication

    # Step 2.5: Generate agent class
    if args.generate_vocab_enabled or args.run_all:
        with logger.contextualize(stage="generate_agent_class"):
            generate_agent_class(config)

    # Step 3: Initial Filtering
    if args.init_filter_enabled or args.run_all:
        with logger.contextualize(stage="initial_filter"):
            run_initial_filters(config)

    # Step 4: Train/Val/Test Split (75:5:20)
    if args.split_enabled or args.run_all:
        with logger.contextualize(stage="split"):
            split_by_document(config)

    # Step 5: Stage-Specific Filtering
    if args.stage1_filter or args.run_all:
        with logger.contextualize(stage="stage1"):
            run_stage1_filters(config)

    if args.stage2_filter or args.run_all:
        with logger.contextualize(stage="stage2"):
            run_stage2_filters(config)

    if args.stage3_filter or args.run_all:
        with logger.contextualize(stage="stage3"):
            run_stage3_filters(config)

    if args.stage4_filter or args.run_all:
        with logger.contextualize(stage="stage4"):
            run_stage4_filters(config)


if __name__ == "__main__":
    main()
