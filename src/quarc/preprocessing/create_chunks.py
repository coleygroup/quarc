import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from loguru import logger


def collect_records_from_json(filepath):
    """Collect records from json file. Each line is a reaction record."""
    with open(filepath, "r") as f:
        for line in f:  # each line is a record
            yield json.loads(line)


def save_init_chunk(output_dir, chunk, chunk_number, chunk_source, chunk_year):
    print(f"saving {chunk_source}_{chunk_year}_chunk_{chunk_number}.pkl")
    with open(
        os.path.join(output_dir, f"{chunk_source}_{chunk_year}_chunk_{chunk_number}.pkl"), "wb"
    ) as f:
        pickle.dump(chunk, f)


def load_init_chunk(path):
    try:
        with open(path, "rb") as f:
            print("loading", str(path).split("/")[-2], str(path).split("/")[-1])
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading chunk {path}:{e}")
        return []


def save_grouped_chunk(data, folder, idx):
    filename = os.path.join(folder, f"grouped_chunk_{idx}.pkl")
    print(f"saving grouped chunk {idx}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def create_data_chunk_from_year(year_dir_path, init_output_dir, chunk_size):
    chunk = []
    chunk_count = 0
    chunk_source, chunk_year = year_dir_path.split("/")[-2:]
    print(f"Starting: {chunk_source} {chunk_year}", flush=True)
    for file in os.listdir(year_dir_path):
        file_path = os.path.join(year_dir_path, file)
        for record in collect_records_from_json(file_path):
            chunk.append(record)

            if len(chunk) == chunk_size:
                save_init_chunk(init_output_dir, chunk, chunk_count, chunk_source, chunk_year)
                chunk_count += 1
                chunk = []

    if chunk:  # Save the last chunk
        save_init_chunk(init_output_dir, chunk, chunk_count, chunk_source, chunk_year)
    print(
        f"Finished processing: {chunk_source} {chunk_year}, total {chunk_count} chunks", flush=True
    )


def create_initial_chunks(config: dict):
    """Create chunks from raw JSON files"""
    logger.info("---Creating initial chunks---")
    db_dir = Path(config["chunking"]["raw_input_dir"])  # pistachio raw json
    init_chunks_output_dir = Path(config["chunking"]["initial_chunks_dir"])  # initial chunks
    init_chunks_output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = config["chunking"]["chunk_size"]
    num_workers = config["chunking"]["num_workers"]

    tasks = []  # each task is a source/year directory with only json files
    for source in os.listdir(db_dir):
        source_path = os.path.join(db_dir, source)
        for year in sorted(os.listdir(source_path), reverse=True):
            tasks.append((os.path.join(source_path, year), init_chunks_output_dir, chunk_size))

    logger.info(f"Creating {len(tasks)} initial chunks")
    with Pool(num_workers) as pool:
        pool.starmap(create_data_chunk_from_year, tasks)


def regroup_chunks(config: dict):
    """Group initial chunks {source}_{year}_chunk_{idx}.pkl to larger chunks"""
    logger.info("---Regrouping chunks---")
    init_chunks_dir = Path(config["chunking"]["initial_chunks_dir"])
    grouped_chunks_dir = Path(config["chunking"]["grouped_chunks_dir"])
    grouped_chunks_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config["chunking"]["group_batch_size"]
    chunk_size = config["chunking"]["chunk_size"]
    num_workers = config["chunking"]["num_workers"]

    all_chunks = list(init_chunks_dir.glob("*.pkl"))

    grouped_data = []
    chunk_idx = 0
    total_processed = 0

    # parallelize reading in batch
    with Pool(num_workers) as pool:
        batch_start = 0
        while batch_start < len(all_chunks):
            batch_files = all_chunks[batch_start : batch_start + batch_size]
            results = pool.map(
                load_init_chunk, batch_files
            )  # if initial chunk over threshold, will need split first

            for data in results:
                grouped_data.extend(data)
                total_processed += len(data)

                while len(grouped_data) >= chunk_size:
                    save_grouped_chunk(grouped_data[:chunk_size], grouped_chunks_dir, chunk_idx)
                    grouped_data = grouped_data[chunk_size:]
                    chunk_idx += 1

            batch_start += batch_size
            print(f"Processed {total_processed} records", flush=True)

    if grouped_data:  # saving any remaining data
        save_grouped_chunk(grouped_data, grouped_chunks_dir, chunk_idx)
