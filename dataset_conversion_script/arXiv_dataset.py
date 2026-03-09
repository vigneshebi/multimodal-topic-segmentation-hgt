# ==========================================================
# convert_arxiv.py
# Convert arXiv papers → unified format
# ==========================================================

import pandas as pd


def convert_arxiv(input_file, output_file):

    data = pd.read_json(input_file, lines=True)

    rows = []

    for _, paper in data.iterrows():

        doc_id = paper["id"]

        text = paper["abstract"]

        sentences = text.split(".")

        for i, sent in enumerate(sentences):

            rows.append({
                "doc_id": doc_id,
                "unit_id": f"U{i+1}",
                "temporal_index": i + 1,
                "content": sent.strip(),
                "modality": "text"
            })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("arXiv conversion completed")


if __name__ == "__main__":

    convert_arxiv(
        "datasets_raw/arxiv/arxiv.json",
        "datasets_processed/arxiv_standard.csv"
    )