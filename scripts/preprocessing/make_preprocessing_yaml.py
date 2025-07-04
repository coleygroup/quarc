import os
import argparse


def make_yaml(data_directory, output_directory, chunk_size=50, delete_if_exists=False):
    if delete_if_exists:
        if os.path.exists(output_directory):
            os.system(f"rm -r {output_directory}")

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    else:
        print(f"Directory {output_directory} already exists.")
        return

    mkdirs = [
        f"{output_directory}/logs",
        f"{output_directory}/logs/preprocessing",
        f"{output_directory}/data",
        f"{output_directory}/data/interim",
        f"{output_directory}/data/interim/23Q2_dump",
        f"{output_directory}/data/interim/23Q2_grouped",
        f"{output_directory}/data/interim/temp_dedup",
        f"{output_directory}/data/interim/split",
        f"{output_directory}/data/processed",
        f"{output_directory}/data/processed/agent_encoder",
        f"{output_directory}/data/processed/stage1",
        f"{output_directory}/data/processed/stage2",
        f"{output_directory}/data/processed/stage3",
        f"{output_directory}/data/processed/stage4",
    ]

    for d in mkdirs:
        if not os.path.exists(d):
            os.mkdir(d)
        else:
            print(f"Directory {d} already exists.")
            return

    wrte = f"""dirs:
    raw_dir: &raw_dir "{data_directory}" # Raw JSON files from Pistachio database
    log_dir: &log_dir "{output_directory}/logs/preprocessing" # Preprocessing logs and debug information

    # Intermediate data directories
    dump_dir: &dump_dir "{output_directory}/data/interim/23Q2_dump" # Initial chunks from raw JSON, one reaction dictionary per line
    grouped_dir: &grouped_dir "{output_directory}/data/interim/23Q2_grouped" # Regrouped larger chunks for efficient processing
    temp_dedup_dir: &temp_dedup_dir "{output_directory}/data/interim/temp_dedup" # Temporary storage for locally deduplicated chunks
    split_dir: &split_dir "{output_directory}/data/interim/split" # Train/val/test splits by document ID (full_train.pickle, full_val.pickle, full_test.pickle)

    # Key intermediate files
    final_dedup_path: &final_dedup_path "{output_directory}/data/interim/23Q2_final_deduped.pickle" # Globally deduplicated reactions
    final_dedup_filtered_path: &final_dedup_filtered_path "{output_directory}/data/interim/23Q2_final_deduped_filtered.pickle" # After initial filtering (length, atom count, etc.)

    # Processed data directories and files
    agent_encoder_list_path: &agent_encoder_list_path "{output_directory}/data/processed/agent_encoder/agent_encoder_list.json" # Agent vocabulary list
    agent_other_dict_path: &agent_other_dict_path "{output_directory}/data/processed/agent_encoder/agent_other_dict.json" # Rare agent mappings (e.g., other_Pd)
    conv_rules_path: &conv_rules_path "{output_directory}/data/processed/agent_encoder/agent_rules_v1.json" # Agent standardization rules

    # Stage-specific filtered data
    stage1_dir: &stage1_dir "{output_directory}/data/processed/stage1" # Agent data
    stage2_dir: &stage2_dir "{output_directory}/data/processed/stage2" # Temperature data
    stage3_dir: &stage3_dir "{output_directory}/data/processed/stage3" # Reactant amount data
    stage4_dir: &stage4_dir "{output_directory}/data/processed/stage4" # Agent amount data

logging:
    log_dir: *log_dir
    debug_logging: false # detailed logging, failed reactions saved
    debug_stages:
        chunking: true
        data_collection: true
        initial_filter: true
        document_split: true
        condition_filter: true

chunking:
    raw_input_dir: *raw_dir
    initial_chunks_dir: *dump_dir
    grouped_chunks_dir: *grouped_dir
    num_workers: 8
    chunk_size: {chunk_size}
    group_batch_size: 10

data_collection:
    input_dir: *grouped_dir
    temp_dedup_dir: *temp_dedup_dir
    output_path: *final_dedup_path

generate_agent_class:
    input_path: *final_dedup_path
    output_encoder_path: *agent_encoder_list_path # agent vocab
    output_other_dict_path: *agent_other_dict_path # other_{'{Pd/Rh/...}'}
    conv_rules_path: *conv_rules_path
    minocc: 50
    metal_minocc: 50

initial_filter:
    input_path: *final_dedup_path
    output_path: *final_dedup_filtered_path
    length_filters:
        product:
            min: 1
            max: 1
        reactant:
            min: 1
            max: 5
        agent:
            min: 0
            max: 5
    atom_filters:
        max_reactant_atoms: 50
        max_product_atoms: 50

document_split:
    input_path: *final_dedup_filtered_path
    output_dir: *split_dir
    split_ratios:
        train: 0.75
        val: 0.05
        test: 0.20
    seed: 42
    save_split_info: true # Save document IDs used in each split

# Agent Filters
stage_1_filter:
    input_dir: *split_dir
    output_dir: *stage1_dir

# Temperature Filters
stage_2_filter:
    input_dir: *split_dir
    output_dir: *stage2_dir
    temperature_range: # in Celsius
        lower: -100
        upper: 200

# Reactant Amount Filters
stage_3_filter:
    input_dir: *split_dir
    output_dir: *stage3_dir
    ratio_range:
        lower: 0.001
        upper: 7.0

# Agent Amount Filters
stage_4_filter:
    input_dir: *split_dir
    output_dir: *stage4_dir
    ratio_range:
        lower: 0.001
        upper: 1000.0
        """
    with open(f"{output_directory}/preprocessing.yaml", "w") as f:
        f.write(wrte)


def main():
    parser = argparse.ArgumentParser(
        description="Create a preprocessing.yaml file for the cond_rec_clean project."
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory where the raw data is stored.",
        required=True,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The directory where the preprocessing.yaml will be saved.",
        default="output",
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="The size of the chunks to be created.",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--delete_if_exists",
        action="store_true",
        help="Delete the output directory if it exists.",
        default=False,
        required=False,
    )
    args = parser.parse_args()
    make_yaml(args.data_directory, args.output_directory, args.chunk_size, args.delete_if_exists)


if __name__ == "__main__":
    main()
