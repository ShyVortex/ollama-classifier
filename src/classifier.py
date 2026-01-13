import json
import os
import sys
import random

import pandas as pd
import requests
import argparse

this_dir = os.path.dirname(os.path.abspath(__file__))
text_column = 'review'
output_path = os.path.normpath(os.path.join(this_dir, '..', 'output'))
output_file = os.path.join(output_path, 'classified_reviews.jsonl')

ollama_api = "http://localhost:11434/api/generate"

# Stable Diffusion seed range (for reference)
min_val = -9223372036854775808
max_val = 18446744073709551615

# Expected categories in output
valid_answers = ["BUG", "FEATURE", "SECURITY", "PERFORMANCE", "USABILITY", "ENERGY", "OTHER"]


def call_ollama(model, text, rel_syspath, reasoning):
    """Sends request to Ollama and returns response."""
    abs_syspath = os.path.normpath(os.path.join(this_dir, '..', rel_syspath))
    sys_prompt = open(abs_syspath, 'r').read()

    input_prompt = f"Categorize: {text}"

    payload = {
        "model": model,
        "prompt": input_prompt,
        "system": sys_prompt,
        "stream": False,         # False = we wait for the complete answer,
        "format": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string"
                },
                "category": {
                    "type": "string",
                    "enum": valid_answers
                }
            },
            "required": ["category"]
        } if reasoning
        else {
            "type": "string",
            "enum": valid_answers
        },
        "options": {
            "temperature": 0.0,                         # low temperature leads to more technical answers
            "seed": random.randint(min_val, max_val),   # for reproducibility
            "num_predict": 128 if reasoning else 5,     # max predicted tokens
            "num_ctx": 4096,                            # enough context for a review
            "stop": ["\n", "."],                        # stop the model on new line or period
            "top_k": 10,                                # reduce probability of generating gibberish
            "top_p": 0.5,                               # value for more or less varied output
        },
        "required": ["analysis", "category"] if reasoning else None
    }

    try:
        response = requests.post(ollama_api, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"\n[API Failure] {e}")
        sys.exit(1)


def get_processed_count():
    """Counts how many rows have been written in the output file."""
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        return 0
    elif not os.path.exists(output_file):
        return 0
    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)


def main(data, model, prompt, reasoning):
    # 1. Handle arguments and edge cases
    try:
        if model is None or prompt is None or reasoning is None:
            raise ValueError()
        if reasoning == 'Y':
            reasoning = True
        elif reasoning == 'N':
            reasoning = False
        else:
            raise ValueError()
    except ValueError:
        print_usage()
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Prompt path: {prompt}")
    print(f"Use reasoning: " + ("Yes" if reasoning else "No") + "\n")

    # 2. Dataset loading
    print(f"Loading data from {data}...")
    try:
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.endswith('.xlsx'):
            df = pd.read_excel(data)
        else:
            print("Unsupported file format. Use CSV or XLSX.")
            return
    except Exception as e:
        print(f"Error while opening file: {e}")
        return

    total_rows = len(df)

    # 3. Resume check
    processed_count = get_processed_count()

    if processed_count > 0:
        print(f"Found pre-existing output file with {processed_count} instances.")
        if processed_count >= total_rows:
            print("All rows have already been analyzed!")
            return
        print(f"Resuming analysis from row {processed_count + 1}...\n")
    else:
        print("No output file found. Starting from scratch...\n")

    # 4. Processing loop
    ## Dataframe slicing: we skip rows already processed
    df_to_process = df.iloc[processed_count:]

    for index, row in df_to_process.iterrows():
        current_step = hash(index) + 1
        text_content = str(row[text_column])

        print(f"Analysis in progress... [{current_step}/{total_rows}]", end="\n")

        llm_response = call_ollama(model, text_content, prompt, reasoning)

        if llm_response:
            clean_category = llm_response.upper().replace('"', '').strip()

            # If result is not valid even after cleaning, we classify it as ERROR
            final_category = clean_category if clean_category in valid_answers else "ERROR"

            if reasoning:
                analysis = json.load(llm_response)
                ext_thinking = []

                for item in analysis:
                    ext_thinking.append(item['analysis'])

                result_obj = {
                    "id": current_step,
                    "text": text_content,
                    "analysis": ext_thinking,
                    "category": final_category,
                }
            else:
                result_obj = {
                    "id": current_step,
                    "category": final_category,
                }

            # 5. We append the result immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result_obj, f, ensure_ascii=False)
                f.write('\n')  # new row for JSONL format
        else:
            print(f"\nError thrown in line {current_step}, row skipped.")

    print(f"\nAnalysis complete! Data has been saved to {output_file}")


def print_usage():
    print("ERROR: Invalid program execution\n")
    print("Basic usage: python classifier.py --prompt [PROMPT_PATH]")
    print("Complete usage: python classifier.py --data [DATASET_PATH] --model [MODEL_NAME]"
          "--prompt [PROMPT_PATH] --reasoning [Y/N]"
          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review Classification Script via local LLMs")

    parser.add_argument(
        "--data",
        type=str,
        default=os.path.normpath(os.path.join(this_dir, '..', 'data', 'rq1-data-nc.csv')),
        help="Path to the input dataset (must be in CSV or XLSX format)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name to use (e.g. llama3.2, deepseek-r1)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Path to a system prompt text file"
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        default="N",
        help="Use model reasoning or not (if available)"
    )

    # Handle no args passed
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    args = parser.parse_args()

    main(args.data, args.model, args.prompt, args.reasoning)