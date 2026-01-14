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
output_file = ""

ollama_api = "http://localhost:11434/api/generate"

# Stable Diffusion seed range (for reference)
min_val = -9223372036854775808
max_val = 18446744073709551615

# Expected categories in output
valid_answers = ["BUG", "FEATURE", "SECURITY", "PERFORMANCE", "USABILITY", "ENERGY", "OTHER"]


def call_ollama(model, text, rel_syspath, reasoning):
    """Sends request to Ollama and returns response."""

    # If given path is valid already, use it (may already be absolute)
    if os.path.exists(rel_syspath):
        abs_syspath = os.path.abspath(rel_syspath)
    else:
        # Otherwise, assume it's placed in the 'prompts' folder
        abs_syspath = os.path.normpath(os.path.join(this_dir, '..', rel_syspath))

    try:
        with open(abs_syspath, 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found at: {abs_syspath}")
        sys.exit(1)

    input_prompt = f"Categorize: {text}"

    payload = {
        "model": model,
        "prompt": input_prompt,
        "system": sys_prompt,
        "stream": False,   # We don't need text to be rendered immediately as soon as it's produced
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
            "required": ["analysis", "category"]
        } if reasoning
        else {
            "type": "string",
            "enum": valid_answers
        },
        "think": True if reasoning else False, # refer to https://docs.ollama.com/capabilities/thinking#thinking
        "options": {
            "temperature": 0.0,                         # low temperature leads to more technical answers
            "seed": random.randint(min_val, max_val),   # for reproducibility
            "num_predict": 512 if reasoning else 5,     # max predicted tokens
            "num_ctx": 4096,                            # enough context for a review
            "stop": None if reasoning else ["\n", "."], # stop model on new line or period if not reasoning
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
        return None


def get_processed_rows(model):
    """
    Reads the output file and returns a set of row IDs that have been
    successfully processed, excluding errors or undefined categories.
    """
    global output_file
    output_file = os.path.join(output_path, f"{model}_classification.jsonl")

    processed_rows = set()
    retry_labels = ["ERROR", "UNDEFINED"]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        category = data.get("category", "ERROR")

                        # If classification didn't fail, add row to processed list
                        if category not in retry_labels:
                            processed_rows.add(data['id'])

                except json.JSONDecodeError:
                    continue
    return processed_rows


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
    processed_rows = get_processed_rows(model)
    print(f"Already completed: {len(processed_rows)}/{total_rows} rows.\n")

    if len(processed_rows) >= total_rows:
        print("All rows appear to be successfully categorized!")
        # Remove the return if you want to force an additional check
        return

    # 4. Processing loop
    for index, row in df.iterrows():
        current_step = index + 1

        # --- SMART SKIP ---
        if current_step in processed_rows:
            continue
        # ------------------

        text_content = str(row[text_column])

        print(f"Analysis in progress... [{current_step}/{total_rows}]", end="\n")

        llm_response = call_ollama(model, text_content, prompt, reasoning)

        if llm_response:
            final_category = "EMPTY"
            result_obj = {
                "id": current_step,
                "category": final_category,
            }

            if reasoning:
                try:
                    analysis = json.loads(llm_response)

                    final_category = analysis.get("category", "EMPTY").upper()
                    ext_thinking = analysis.get("analysis", "")

                    result_obj = {
                        "id": current_step,
                        "text": text_content,
                        "analysis": ext_thinking,
                        "category": final_category,
                    }
                except json.JSONDecodeError:
                    print(f"\n[JSON Error] Could not parse: {llm_response[:50]}...")
                    result_obj["category"] = "ERROR"
            else:
                clean_category = llm_response.upper().replace('"', '').strip()

                # If result is not valid even after cleaning, we classify it as UNDEFINED
                final_category = clean_category if clean_category in valid_answers else "UNDEFINED"
                result_obj = {
                    "id": current_step,
                    "category": final_category,
                }

            # Final validation of category
            if final_category not in valid_answers and final_category == "EMPTY":
                final_category = "UNDEFINED"

            result_obj["category"] = final_category

            # 5. We append the result immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result_obj, f, ensure_ascii=False)
                f.write('\n')  # new row for JSONL format
        else:
            print(f"\nError thrown in line {current_step}, row skipped.")

    print(f"\nAnalysis complete! Data has been saved to {output_file}")


def print_usage():
    print("ERROR: Invalid program execution\n")
    print("Basic usage: python classifier.py --model [MODEL_NAME] --prompt [PROMPT_PATH]")
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