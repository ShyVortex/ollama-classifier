import argparse
import os
import sys
import json
import pandas as pd


this_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(this_dir, '..'))

data_dir = os.path.join(project_root, 'data')
predictions_root = os.path.join(data_dir, 'predictions')
output_dir = os.path.join(data_dir, 'output')

output_file = os.path.join(output_dir, 'comparison_result.json')


def main(clap_file, model_family, model_file, prediction_source):

    os.makedirs(output_dir, exist_ok=True)

    clap_path = os.path.join(data_dir, clap_file)

    try:
        clap_df = pd.read_csv(clap_path)
    except Exception as e:
        print(f"ERROR: Cannot open CLAP file: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = {"id", "category"}
    if not required_cols.issubset(clap_df.columns):
        print(
            "ERROR: The provided CLAP dataset must contain "
            "'id' and 'category' columns."
        )
        sys.exit(1)

    # Keep only what we need
    clap_df = clap_df[["id", "category"]]

    # Normalize
    clap_df["id"] = clap_df["id"].astype(int)
    clap_df["category"] = clap_df["category"].astype(str).str.strip().str.upper()

    clap_dict = dict(zip(clap_df["id"], clap_df["category"]))

    model_path = os.path.join(predictions_root, prediction_source, model_family, model_file)

    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Cannot open model prediction file: {e}")
        sys.exit(1)

    model_dict = {}
    for item in model_data:
        try:
            rid = int(item["id"])
            cat = str(item["category"]).strip().upper()
            model_dict[rid] = cat
        except Exception:
            continue  # robustness by design

    results = []
    matches = 0

    for rid, clap_cat in clap_dict.items():
        model_cat = model_dict.get(rid)

        match = clap_cat == model_cat
        if match:
            matches += 1

        results.append({
            "id": rid,
            "category": model_cat,
            "match": match
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    accuracy = matches / total

    print("=== EVALUATION REPORT ===")
    print(f"CLAP file: {clap_file}")
    print(f"Model family: {model_family}")
    print(f"Model file: {model_file}")
    print(f"Istanze: {total}")
    print(f"Match: {matches}")
    print(f"Mismatch: {total - matches}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare CLAP ground truth with model predictions"
    )

    parser.add_argument(
        "--clap",
        type=str,
        required=True,
        help="CLAP CSV filename inside the data directory"
    )

    parser.add_argument(
        "--model-family",
        type=str,
        required=True,
        help="Model folder inside data/predictions (e.g. Gemini-3.5, GPT-5.2)"
    )

    parser.add_argument(
        "--prediction-source",
        type=str,
        required=True,
        choices=["dataset", "sample"],
        help="Prediction source inside data/predictions (dataset | sample)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model prediction JSON filename"
    )

    args = parser.parse_args()

    main(args.clap, args.model_family, args.model, args.prediction_source)
