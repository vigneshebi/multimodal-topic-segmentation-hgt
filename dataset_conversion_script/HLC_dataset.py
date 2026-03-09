# ==========================================================
# generate_synthetic_hlc.py
# Generate synthetic heterogeneous lecture dataset
# ==========================================================

import pandas as pd
import random


modalities = ["text", "equation", "table", "diagram"]


def generate_dataset(num_docs=100):

    rows = []

    for d in range(num_docs):

        doc_id = f"lecture_{d}"

        for i in range(20):

            modality = random.choice(modalities)

            rows.append({
                "doc_id": doc_id,
                "unit_id": f"U{i+1}",
                "temporal_index": i + 1,
                "content": f"sample content {i}",
                "modality": modality
            })

    df = pd.DataFrame(rows)

    df.to_csv("datasets_processed/synthetic_hlc_standard.csv", index=False)


if __name__ == "__main__":

    generate_dataset()