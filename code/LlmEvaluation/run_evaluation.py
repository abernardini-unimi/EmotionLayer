import pandas as pd
import time
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from provider.hugging_face import hf_inference
from provider.openai import openai_inference
from provider.claude import claude_inference
from provider.gemini import gemini_inference
from provider.groq import groq_inference

from config.config import OUTPUT_EXCEL, MODEL_FILE, PROMPT_FILE_PATH, DATASET_FILE, LLM_OUTPUT_JSON
from utils import load_json_from_file, load_text_file, generate_report

# Provider-to-function mapping
PROVIDER_MAPPING = {
    "HuggingFace": hf_inference,
    "OpenAI": openai_inference,
    "Anthropic": claude_inference,
    "Gemini": gemini_inference,
    "Groq": groq_inference
}

def process_single_inference(model_row, case, prompt_template):
    """
    Executes a single inference (Thread worker)
    """
    model_name = model_row['model_name']
    provider_name = model_row['provider']

    try:
        price_input_1m = float(model_row['input'])
        price_output_1m = float(model_row['output'])
    except (ValueError, TypeError):
        price_input_1m = 0.0
        price_output_1m = 0.0

    inference_func = PROVIDER_MAPPING.get(provider_name)

    if not inference_func:
        return None

    case_input_str = json.dumps(case['input'], ensure_ascii=False)

    try:
        result_data = inference_func(
            model_name=model_name,
            system_prompt=prompt_template,
            user_input=case_input_str
        )

        response_content = result_data.get("response", "")
        latency = result_data.get("latency", 0.0)
        in_tokens = result_data.get("input_tokens", 0)
        out_tokens = result_data.get("output_tokens", 0)

        # Cost calculation
        cost_input = (in_tokens / 1_000_000) * price_input_1m
        cost_output = (out_tokens / 1_000_000) * price_output_1m
        total_cost = cost_input + cost_output

        return {
            "model_name": model_name,
            "data": {
                "provider": provider_name,
                "response_raw": response_content,
                "metrics": {
                    "latency_s": round(latency, 4),
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "cost_usd": round(total_cost, 7)
                }
            }
        }

    except Exception as e:
        print(f"❌ Thread error ({model_name}): {e}")
        return {
            "model_name": model_name,
            "data": {
                "provider": provider_name,
                "response_raw": str(e),
                "metrics": {"latency_s": 0, "cost_usd": 0}
            }
        }


def main(debug_mode=False, max_workers=5):
    print('*** STARTING LLM ARCHITECTURE BENCHMARK (INTEGRATED REPORT) ***')
    start_total = time.time()

    prompt_template = load_text_file(PROMPT_FILE_PATH)
    if not prompt_template:
        print("❌ Unable to load prompt file.")
        return

    dataset = load_json_from_file(DATASET_FILE)
    if not dataset:
        print("❌ Unable to load dataset.")
        return

    if debug_mode:
        dataset = dataset[:1]
        print("⚠️ DEBUG MODE: Processing 1 case only.")

    try:
        models_df = pd.read_excel(MODEL_FILE)
        models_df.columns = models_df.columns.str.strip().str.lower()
        if 'input_' in models_df.columns:
            models_df.rename(columns={'input_': 'input'}, inplace=True)
    except Exception as e:
        print(f"❌ Model Excel loading error: {e}")
        return

    print(f"✅ Loaded: {len(dataset)} cases and {len(models_df)} models.")
    print(f"🚀 Starting with {max_workers} threads...")

    results_storage = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for case in dataset:
            case_id = case.get('id', 'unknown')
            print(f"\n🔹 Processing Case: {case_id}")

            future_to_model = {}
            model_results = {}

            for _, row in models_df.iterrows():
                future = executor.submit(process_single_inference, row, case, prompt_template)
                future_to_model[future] = row['model_name']

            for future in as_completed(future_to_model):
                result = future.result()
                if result:
                    model_results[result["model_name"]] = result["data"]

            results_storage.append({
                "test_case_id": case_id,
                "emotion_detected": case.get('emotion_detected', 'unknown'),
                "target_strategy": case.get('target_strategy', 'N/A'),
                "original_input": case['input'],
                "benchmark_results": model_results
            })

    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save JSON backup
    with open(LLM_OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_storage, f, indent=2, ensure_ascii=False)
    print(f"\n📂 JSON backup saved: {LLM_OUTPUT_JSON}")

    # Save Excel report
    generate_report(results_storage, OUTPUT_EXCEL)

    elapsed = time.time() - start_total
    print(f"\n⏱️ Total benchmark time: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Runner")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 1 case)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel threads (default: 5)"
    )

    args = parser.parse_args()

    main(
        debug_mode=args.debug,
        max_workers=args.max_workers
    )