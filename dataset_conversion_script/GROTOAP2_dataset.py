# ==========================================================
# convert_grotoap2.py
# Convert GROTOAP2 dataset → unified format
# ==========================================================

import os
import pandas as pd


def convert_grotoap2(input_folder, output_file):

    rows = []

    for file in os.listdir(input_folder):

        doc_id = file.replace(".txt", "")

        path = os.path.join(input_folder, file)

        with open(path, "r", encoding="utf-8") as f:

            blocks = f.readlines()

        for i, block in enumerate(blocks):

            rows.append({
                "doc_id": doc_id,
                "unit_id": f"U{i+1}",
                "temporal_index": i + 1,
                "content": block.strip(),
                "modality": "text"
            })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("GROTOAP2 conversion completed")


if __name__ == "__main__":

    convert_grotoap2(
        "datasets_raw/grotoap2/",
        "datasets_processed/grotoap2_standard.csv"
    )