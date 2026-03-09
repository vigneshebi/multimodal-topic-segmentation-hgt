# ==========================================================
# prepare_all_datasets.py
# Master script to prepare all datasets for experiments
# ==========================================================

import os
import pandas as pd

from convert_lecturebank import convert_lecturebank
from convert_arxiv import convert_arxiv
from convert_docbank import convert_docbank
from convert_s2orc import convert_s2orc
from convert_grotoap2 import convert_grotoap2


# ==========================================================
# Configuration
# ==========================================================

RAW_DATASETS = {
    "lecturebank": "datasets_raw/lecturebank/lecturebank.json",
    "arxiv": "datasets_raw/arxiv/arxiv.json",
    "docbank": "datasets_raw/docbank/",
    "s2orc": "datasets_raw/s2orc/s2orc.json",
    "grotoap2": "datasets_raw/grotoap2/"
}

PROCESSED_FOLDER = "datasets_processed/"
FINAL_DATASET = "datasets_processed/final_multimodal_dataset.csv"


# ==========================================================
# Step 1: Convert Individual Datasets
# ==========================================================

def run_all_conversions():

    print("Starting dataset conversions...")

    convert_lecturebank(
        RAW_DATASETS["lecturebank"],
        PROCESSED_FOLDER + "lecturebank_standard.csv"
    )

    convert_arxiv(
        RAW_DATASETS["arxiv"],
        PROCESSED_FOLDER + "arxiv_standard.csv"
    )

    convert_docbank(
        RAW_DATASETS["docbank"],
        PROCESSED_FOLDER + "docbank_standard.csv"
    )

    convert_s2orc(
        RAW_DATASETS["s2orc"],
        PROCESSED_FOLDER + "s2orc_standard.csv"
    )

    convert_grotoap2(
        RAW_DATASETS["grotoap2"],
        PROCESSED_FOLDER + "grotoap2_standard.csv"
    )

    print("All dataset conversions completed")


# ==========================================================
# Step 2: Validate Dataset Format
# ==========================================================

def validate_dataset(file_path):

    required_columns = [
        "doc_id",
        "unit_id",
        "temporal_index",
        "content",
        "modality"
    ]

    df = pd.read_csv(file_path)

    for col in required_columns:

        if col not in df.columns:
            raise Exception(f"{file_path} missing column: {col}")

    print(f"{file_path} validation passed")


# ==========================================================
# Step 3: Merge All Datasets
# ==========================================================

def merge_datasets():

    print("Merging all datasets...")

    files = [
        "lecturebank_standard.csv",
        "arxiv_standard.csv",
        "docbank_standard.csv",
        "s2orc_standard.csv",
        "grotoap2_standard.csv"
    ]

    dfs = []

    for f in files:

        path = os.path.join(PROCESSED_FOLDER, f)

        validate_dataset(path)

        df = pd.read_csv(path)

        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    final_df.to_csv(FINAL_DATASET, index=False)

    print("Final dataset created:", FINAL_DATASET)

    print("Total samples:", len(final_df))


# ==========================================================
# Step 4: Dataset Statistics
# ==========================================================

def dataset_statistics():

    df = pd.read_csv(FINAL_DATASET)

    print("\nDataset Statistics")

    print("Total Documents:", df["doc_id"].nunique())

    print("Total Units:", len(df))

    print("\nModality Distribution:")

    print(df["modality"].value_counts())


# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    run_all_conversions()

    merge_datasets()

    dataset_statistics()

    print("\nDataset preparation completed successfully")