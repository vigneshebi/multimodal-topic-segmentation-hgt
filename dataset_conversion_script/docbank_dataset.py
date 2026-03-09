# ==========================================================
# convert_docbank.py
# Convert DocBank dataset → unified format
# ==========================================================

import os
import pandas as pd


def convert_docbank(input_folder, output_file):

    rows = []

    for file in os.listdir(input_folder):

        path = os.path.join(input_folder, file)

        doc_id = file.replace(".txt", "")

        with open(path, "r", encoding="utf-8") as f:

            lines = f.readlines()

        for i, line in enumerate(lines):

            rows.append({
                "doc_id": doc_id,
                "unit_id": f"U{i+1}",
                "temporal_index": i + 1,
                "content": line.strip(),
                "modality": "text"
            })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("DocBank conversion completed")


if __name__ == "__main__":

    convert_docbank(
        "datasets_raw/docbank/",
        "datasets_processed/docbank_standard.csv"
    )