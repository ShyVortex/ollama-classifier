import json
import os
import sys
import random
import re

import GPUtil
import pandas as pd
import requests
import argparse

this_dir = os.path.dirname(os.path.abspath(__file__))
text_column = 'review'
output_path = os.path.normpath(os.path.join(this_dir, '..', 'output'))
output_file = ""
post_analysis = False

ollama_api = "http://localhost:11434/api/generate"

# Stable Diffusion seed range (for reference)
min_val = -9223372036854775808
max_val = 18446744073709551615

# Expected categories in output
valid_answers = ["BUG", "FEATURE", "SECURITY", "PERFORMANCE", "USABILITY", "ENERGY", "OTHER"]

# Initialize hardware-related variables
cpu_threads = 0
v_ram = {}
batch_size = 0


def finalize_json(input_jsonl, output_folder):
    """Takes the streaming JSONL classification and converts it to a JSON file."""

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_jsonl))[0]

    f_path = os.path.join(output_folder, f"{base_name}.json")

    if os.path.exists(f_path):
        print("Finalized JSON file already exists.\n\nRepeating process...")
        os.remove(f_path)

    data = []
    with open(input_jsonl, 'r') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue

    with open(f_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"[SUCCESS] Converted {base_name}.jsonl to {base_name}.json\n")

    if post_analysis:
        print(f"Conversion complete! Data has been saved to {f_path}")


def robust_parse(row_id, snippet, llm_output):
    """
    Attempts to parse a JSON file in a more robust way.
    If json.loads() fails, it tries to clean the string or use regex.
    """

    clean_text = llm_output.replace("```json", "").replace("```", "").strip()

    # Standard attempt
    try:
        reconstructed_obj = json.loads(clean_text)
        category = reconstructed_obj.get("category", "EMPTY").upper()
        thought_process = reconstructed_obj.get("analysis", "")
        return {
            "id": row_id,
            "text_snippet": snippet,
            "analysis": thought_process,
            "category": category
        }
    except json.JSONDecodeError:
        pass

    # Plan B: we try regex extraction
    data = {
        "id": row_id,
        "text_snippet": snippet,
        "analysis": "",
        "category": "EMPTY"
    }

    # Search for "category": "VALUE" in the text
    cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', clean_text, re.IGNORECASE)
    if cat_match:
        data["category"] = cat_match.group(1)
    else:
        # Fallback: we only search the key word
        for valid in valid_answers:
            if valid in clean_text.upper():
                data["category"] = valid
                break
        if "category" not in data:
            for enum in valid_answers:
                if enum in data:
                    data["category"] = enum
                    break
            else:
                data["category"] = "EMPTY"

    # Search for "analysis": "Reasoning text..."
    ana_match = re.search(r'"analysis"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*})', clean_text, re.DOTALL)
    if ana_match:
        data["analysis"] = ana_match.group(1)
    else:
        # If we can't find the clean analysis, we take the rest
        data["analysis"] = clean_text[:500]  # first 500 chars to avoid issues

    return data


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
            "num_predict": 768 if reasoning else 5,     # max predicted tokens
            "num_ctx": 4096,                            # enough context for a review
            "stop": None if reasoning else ["\n", "."], # stop model on new line or period if not reasoning
            "top_k": 10,                                # reduce probability of generating gibberish
            "top_p": 0.5,                               # value for more or less varied output
            "num_gpu": 256,                             # max value for open-webui
            "num_cpu": cpu_threads,                     # use all threads for best possible speed
            "num_batch": batch_size                     # best-fit batch size for GPU
        },
        "required": ["analysis", "category"] if reasoning else None
    }

    try:
        response = requests.post(ollama_api, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"[API Failure] {e}\n")
        if "Not Found for url" in e.args[0]:
            print("The model you chose was not found in Ollama's library. Please try a different one.")
            print("For the full list of available models, refer to: https://ollama.com/library")
            sys.exit(1)
        if reasoning and "Bad Request for url" in e.args[0]:
            print(f"Reasoning not supported for the model {model.upper()}. Please try a different one.")
            print("For the list of available reasoning models, refer to: "
                  "https://ollama.com/search?c=thinking")
            sys.exit(1)
        return None


def get_gpu_memory():
    """Returns both the total and available GPU memory."""

    gpus = GPUtil.getGPUs()

    if not gpus:
        print("No GPUs detected.", file=sys.stderr)
        return None

    all_memory = {gpu.id: gpu.memoryFree for gpu in gpus}

    # Find GPU with the most free memory
    best_gpu_id = max(all_memory, key=all_memory.get)

    print(f"Detected GPU with {all_memory[best_gpu_id]} MB free memory.")
    return all_memory, best_gpu_id


def calc_batchsize():
    """Computes batch dimension to determine how many tokens are processed at once,
    based on available GPU memory."""

    global v_ram
    all_mem, best_gpu_id = get_gpu_memory()
    free_mem = all_mem[best_gpu_id]

    match free_mem:
        case memory if memory <= 6144:
            return 256
        case memory if 6144 < memory <= 8192:
            return 512
        case memory if 8192 < memory <= 12288:
            return 1024
        case memory if 12288 <= memory <= 16384:
            return 2048
        case memory if memory > 16384:
            return 4096
        case _:
            return 512


def reg_check(data_obj):
    """
    Extracts text from JSON output and validates the category.
    If either no category or an invalid category is found it returns False, otherwise True.
    """

    clean_text = data_obj.replace("```json", "").replace("```", "").strip()

    # Search for "category": "VALUE" in the text
    cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', clean_text, re.IGNORECASE)
    if cat_match:
        return True
    else:
        # Fallback: we only search the key word
        for valid in valid_answers:
            if valid in clean_text.upper():
                return True
        if "category" not in data_obj:
            return False

    return False


def get_processed_rows(model, reasoning):
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
                            if reasoning:
                                if "analysis" in data and "text_snippet" in data:
                                    processed_rows.add(data['id'])
                            else:
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
    processed_rows = get_processed_rows(model, reasoning)
    print(f"Already completed: {len(processed_rows)}/{total_rows} rows.\n")

    if len(processed_rows) >= total_rows:
        print("All rows appear to have been categorized...")
        global post_analysis
        post_analysis = True
        finalize_json(output_file, output_path)
        # Remove the return if you want to force an additional check
        return

    # 4. Get hardware info
    global cpu_threads, v_ram, batch_size
    cpu_threads = os.cpu_count()
    print(f"Detected CPU with {cpu_threads} threads.")

    batch_size = calc_batchsize()
    print(f"Using batch size of {batch_size} tokens, based on GPU memory.\n")

    # 5. Processing loop
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
                    snippet = ' '.join(text_content.split()[:5])
                    result_obj = robust_parse(current_step, snippet, llm_response)
                    final_category = result_obj.get("category", "EMPTY")
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

            # 6. We append the result immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result_obj, f, ensure_ascii=False)
                f.write('\n')  # new row for JSONL format
        else:
            print(f"Error thrown in line {current_step}, row skipped.\n")

    # 7. Convert JSONL to JSON
    finalize_json(output_file, output_path)
    print(f"Analysis complete! Data has been saved to {output_file}")


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
        default=os.path.normpath(os.path.join(this_dir, '..', 'data', 'raw_data.csv')),
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