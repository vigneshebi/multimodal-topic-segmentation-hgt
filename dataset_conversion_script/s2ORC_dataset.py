# ==========================================================
# convert_s2orc.py
# Convert S2ORC dataset → unified format
# ==========================================================

import json
import pandas as pd


def convert_s2orc(input_file, output_file):

    rows = []

    with open(input_file, "r") as f:

        for line in f:

            paper = json.loads(line)

            doc_id = paper["paper_id"]

            paragraphs = paper["body_text"]

            for i, p in enumerate(paragraphs):

                rows.append({
                    "doc_id": doc_id,
                    "unit_id": f"U{i+1}",
                    "temporal_index": i + 1,
                    "content": p["text"],
                    "modality": "text"
                })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("S2ORC conversion completed")


if __name__ == "__main__":

    convert_s2orc(
        "datasets_raw/s2orc/s2orc.json",
        "datasets_processed/s2orc_standard.csv"
    )