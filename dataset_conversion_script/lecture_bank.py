# ==========================================================
# convert_lecturebank.py
# Convert LectureBank dataset → unified format
# ==========================================================

import json
import pandas as pd


def convert_lecturebank(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for lecture in data:

        doc_id = lecture["lecture_id"]

        paragraphs = lecture["paragraphs"]

        for i, para in enumerate(paragraphs):

            rows.append({
                "doc_id": doc_id,
                "unit_id": f"U{i+1}",
                "temporal_index": i + 1,
                "content": para,
                "modality": "text"
            })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("LectureBank conversion completed")


if __name__ == "__main__":

    convert_lecturebank(
        "datasets_raw/lecturebank/lecturebank.json",
        "datasets_processed/lecturebank_standard.csv"
    )