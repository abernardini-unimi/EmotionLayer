import json
import random
import re
import pandas as pd
import time
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Provider wrapper and utilities import
from provider.openai import openai_inference
from provider.claude import claude_inference
from utils import load_json_from_file, load_text_file
from config.config import PROMPT_FILE_PATH, JUDGE_PROMPT_PATH, JUDGE_PROVIDER, JUDGE_MODEL, LLM_OUTPUT_JSON, JUDGE_OUTPUT_FILE


# --- UTILITY LOGGING ---
def log(msg, icon="ℹ️"):
    """Print messages with timestamp to track execution."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {msg}")


def save_partial_results(df_eval, filename):
    """Safely save partial results to Excel."""
    try:
        if not os.path.exists(filename):
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_eval.to_excel(writer, sheet_name='Evaluation Details', index=False)
        else:
            with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
                df_eval.to_excel(writer, sheet_name='Evaluation Details', index=False)
        log(f"Incremental save completed on {filename} ({len(df_eval)} total rows).", "💾")
    except Exception as e:
        log(f"Critical error during incremental save: {e}", "❌")


def get_completed_tasks(filename):
    """Read existing Excel file to find already completed tasks (for resume)."""
    completed = set()
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename, sheet_name='Evaluation Details')
            if not df.empty and 'Test Case ID' in df.columns and 'Model' in df.columns:
                for _, row in df.iterrows():
                    # Create a unique ID_Model key
                    completed.add(f"{row['Test Case ID']}_{row['Model']}")
            log(f"Found {len(completed)} tasks already completed in the existing file.", "🔄")
        except Exception as e:
            log(f"Unable to read existing file (it will be overwritten/created): {e}", "⚠️")
    return completed


def _safe_parse_json(content_clean: str) -> dict:
    """Attempt JSON parsing with progressive fallbacks."""
    
    # Attempt 1: direct parsing
    try:
        return json.loads(content_clean)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: extract with regex and retry
    json_match = re.search(r'\{.*\}', content_clean, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Attempt 3: truncate at explanation field and manually close
    # Useful when explanation is the last field and gets truncated
    try:
        # Find the position of the last valid numeric field
        # and rebuild a clean JSON extracting simple fields
        fields = {}
        for key in ['emotion_match', 'strategy_match', 'relevance_score', 
                    'tts_score', 'voice_suitability_score', 'empatic_response']:
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', content_clean)
            if m:
                fields[key] = float(m.group(1))
        
        # Extract explanation until the first issue
        exp_match = re.search(r'"explanation"\s*:\s*"(.*?)(?:"\s*\}|$)', content_clean, re.DOTALL)
        if exp_match:
            # Clean explanation from problematic characters
            explanation = exp_match.group(1)
            explanation = explanation.replace("'", "\\'").rstrip("'\"} \n")
            fields['explanation'] = explanation
        else:
            fields['explanation'] = 'Partial parsing - explanation not extracted'
        
        if fields:
            return fields
    except Exception:
        pass
    
    # Final fallback: default values
    return {
        "emotion_match": 0, "strategy_match": 0, "relevance_score": 0,
        "tts_score": 0, "voice_suitability_score": 0, "empatic_response": 0,
        "explanation": f"Parsing failed. Raw content: {content_clean[:200]}"
    }


def evaluate_single_response(case, model_name, model_result, judge_full_prompt):
    """
    Calls GPT-4o or Claude to evaluate a single response.
    Includes RETRY logic and LONG PAUSE handling for Rate Limit management.
    """
    original_input = case.get('original_input', {})
    user_text = original_input.get('transcription', '')
    user_emotion = original_input.get('vocal_sentiment', [])
    prosody = original_input.get('prosody', {})
    target_strategy = case.get('target_strategy', 'N/A')
    target_emotion = case.get('emotion_detected', 'N/A')
    
    raw_resp = model_result.get('response_raw', '')
    if isinstance(raw_resp, dict):
        raw_resp_str = json.dumps(raw_resp, ensure_ascii=False)
    else:
        raw_resp_str = str(raw_resp)
    
    user_content = f"""
        --- USER SCENARIO TO EVALUATE ---
        User Text: "{user_text}"
        Detected Vocal Emotions: {user_emotion}
        Prosody: {prosody}
        Expected Target Strategy (Ground Truth): {target_strategy}
        Expected Target Emotion (Ground Truth): {target_emotion}
        --- MODEL RESPONSE UNDER REVIEW ({model_name}) ---
        Generated JSON Output: 
        {raw_resp_str}
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Judge Provider Selection
            if JUDGE_PROVIDER.lower() == 'openai':
                evaluation_str = openai_inference(JUDGE_MODEL, judge_full_prompt, user_content)
            elif JUDGE_PROVIDER.lower() == 'claude':
                evaluation_str = claude_inference(JUDGE_MODEL, judge_full_prompt, user_content)
            else:
                return {"explanation": f"Unknown Judge Provider: {JUDGE_PROVIDER}"}

            # Check generic errors returned as string/dict
            if not evaluation_str or (isinstance(evaluation_str, dict) and "error" in evaluation_str):
                 raise ValueError("API Error or Rate Limit (Empty/Error response)")

            # Content extraction
            if isinstance(evaluation_str, dict) and "response" in evaluation_str:
                 evaluation_content = evaluation_str["response"]
            else:
                 evaluation_content = evaluation_str

            # JSON parsing
            if isinstance(evaluation_content, str):
                content_clean = evaluation_content.strip()
                
                match = re.search(r'```json\s*(.*?)\s*```', content_clean, re.DOTALL)
                if match:
                    content_clean = match.group(1).strip()
                elif content_clean.startswith('```'):
                    content_clean = re.sub(r'^```\w*\s*', '', content_clean)
                    content_clean = re.sub(r'\s*```$', '', content_clean).strip()
                
                evaluation = _safe_parse_json(content_clean)
            else:
                evaluation = evaluation_content

            return evaluation

        except Exception as e:
            # Advanced error handling
            err_msg = str(e)
            is_rate_limit = "429" in err_msg or "Rate limit" in err_msg or "quota" in err_msg
            
            if is_rate_limit:
                log(f"🛑 RATE LIMIT on {model_name}! Safety pause 30s...", "⏳")
                time.sleep(30) 
            else:
                wait_time = (5 * (2 ** attempt)) + random.uniform(1, 3)
                if attempt < max_retries - 1:
                    log(f"Error on {model_name} (Attempt {attempt+1}/{max_retries}): {err_msg[:50]}... Waiting {wait_time:.1f}s", "⚠️")
                    time.sleep(wait_time)
                else:
                    log(f"Final failure on {model_name}. Reason: {err_msg}", "❌")
                    return {
                        "emotion_match": 0, "strategy_match": 0, "relevance_score": 0, 
                        "tts_score": 0, "voice_suitability_score": 0, "empatic_response": 0,
                        "explanation": f"API Error after {max_retries} attempts: {err_msg}"
                    }

def main(debug_mode=False, max_workers=1):
    print("\n" + "="*60)
    log(f"LLM-AS-A-JUDGE START | Provider: {JUDGE_PROVIDER} | Model: {JUDGE_MODEL}", "⚖️")
    log(f"Config: {max_workers} Workers | Delay: {2}s | Debug: {debug_mode}", "⚙️")
    print("="*60 + "\n")
    
    # 1. Prompt Loading
    original_prompt_content = load_text_file(PROMPT_FILE_PATH)
    judge_base_instructions = load_text_file(JUDGE_PROMPT_PATH)

    if not original_prompt_content or not judge_base_instructions:
        log("Critical error: Unable to load prompt files.", "❌")
        return

    final_judge_system_prompt = f"""
    {judge_base_instructions}
    
    --- CONTEXT: ORIGINAL SYSTEM PROMPT GIVEN TO THE MODEL ---
    {original_prompt_content}
    ------------------------------------------------------------
    """
    log("Judge prompt successfully assembled.", "✅")
    
    # 2. Dataset Loading
    data = load_json_from_file(LLM_OUTPUT_JSON)
    if not data: 
        log(f"Dataset not found in {LLM_OUTPUT_JSON}", "❌")
        return

    if debug_mode:
        data = data[:1]
        log("DEBUG MODE ENABLED: Analysis limited to the first case.", "🐞")

    # 3. Resume Logic
    completed_tasks = get_completed_tasks(JUDGE_OUTPUT_FILE)
    
    # Load existing data if present, to avoid overwriting
    evaluation_rows = []
    if os.path.exists(JUDGE_OUTPUT_FILE):
        try:
            existing_df = pd.read_excel(JUDGE_OUTPUT_FILE, sheet_name='Evaluation Details')
            evaluation_rows = existing_df.to_dict('records')
        except: pass

    # 4. Scheduling
    log(f"Preparing tasks for {len(data)} test cases...", "🚀")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        skipped_count = 0

        for case in data:
            case_id = case.get('test_case_id')
            target_strategy = case.get('target_strategy')
            emotion_detected = case.get('emotion_detected', 'N/A')
            
            for model_name, res in case.get('benchmark_results', {}).items():
                
                # RESUME CHECK
                task_key = f"{case_id}_{model_name}"
                if task_key in completed_tasks:
                    skipped_count += 1
                    continue

                future = executor.submit(evaluate_single_response, case, model_name, res, final_judge_system_prompt)
                future_map[future] = (case_id, model_name, target_strategy, emotion_detected)
                
                # Pacing to avoid overloading submission
                time.sleep(2)

        if skipped_count > 0:
            log(f"Skipped {skipped_count} tasks already present in the Excel file.", "⏭️")
        
        total_tasks = len(future_map)
        if total_tasks == 0:
            log("All tasks are already completed! Generating final report...", "✨")
        else:
            log(f"Starting execution of {total_tasks} new evaluations...", "▶️")
        
        # 5. Execution and Collection
        completed_now = 0
        
        for future in as_completed(future_map):
            case_id, model_name, target_strategy, emotion_detected = future_map[future]
            completed_now += 1
            
            try:
                eval_result = future.result()
                
                row = {
                    "Test Case ID": case_id,
                    "Target Strategy": target_strategy,
                    "Emotion Detected": emotion_detected,
                    "Model": model_name,
                    "Emotion Match (0/1)": eval_result.get('emotion_match', 0),
                    "Strategy Match (0/1)": eval_result.get('strategy_match', 0),
                    "Relevance (1-5)": eval_result.get('relevance_score', 0),
                    "TTS Alignment (1-5)": eval_result.get('tts_score', 0),
                    "Voice Suitability (1-5)": eval_result.get('voice_suitability_score', 0),
                    "Empatic Response (1-5)": eval_result.get('empatic_response', 0),
                    "Judge Explanation": eval_result.get('explanation', 'N/A')
                }
                evaluation_rows.append(row)
                
                # Progress logging
                percent = int((completed_now / total_tasks) * 100)
                log(f"[{completed_now}/{total_tasks}] {percent}% - Evaluated {case_id} ({model_name})", "📝")
                
                # Save every 5
                if completed_now % 5 == 0:
                    df_temp = pd.DataFrame(evaluation_rows)
                    save_partial_results(df_temp, JUDGE_OUTPUT_FILE)

            except Exception as e:
                log(f"Critical error saving row for {case_id}: {e}", "❌")

    # --- FINAL REPORT GENERATION ---
    print("\n" + "="*60)
    log("Generating Final Report and Pivot Tables...", "📊")
    
    df_eval = pd.DataFrame(evaluation_rows)
    if df_eval.empty: 
        log("No data to save.", "⚠️")
        return

    # Numeric data cleaning
    cols_to_numeric = [
        'Emotion Match (0/1)', 'Strategy Match (0/1)', 'Relevance (1-5)', 
        'TTS Alignment (1-5)', 'Voice Suitability (1-5)', 'Empatic Response (1-5)'
    ]
    for col in cols_to_numeric:
        df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce').fillna(0)

    # Total Score Calculation
    df_eval['Overall Score (100)'] = (
        (df_eval['Emotion Match (0/1)'] * 20) +
        (df_eval['Strategy Match (0/1)'] * 30) + 
        ((df_eval['Relevance (1-5)'] / 5) * 10) + 
        ((df_eval['TTS Alignment (1-5)'] / 5) * 10) + 
        ((df_eval['Voice Suitability (1-5)'] / 5) * 10) +
        ((df_eval['Empatic Response (1-5)'] / 5) * 20)
    )

    # Overall Ranking
    df_rank = df_eval.groupby('Model').agg({
        'Overall Score (100)': 'mean',
        'Emotion Match (0/1)': 'mean',
        'Strategy Match (0/1)': 'mean', 
        'Relevance (1-5)': 'mean',
        'TTS Alignment (1-5)': 'mean',
        'Voice Suitability (1-5)': 'mean',
        'Empatic Response (1-5)': 'mean'
    }).sort_values(by='Overall Score (100)', ascending=False).reset_index().round(2)
    
    # Pivot: Strategy by Strategy
    def format_fraction(x): return f"{int(x.sum())}/{x.count()}"
    # Pivot: Emotion by Strategy
    df_cat_emotion = df_eval.pivot_table(index='Model', columns='Target Strategy', values='Emotion Match (0/1)', aggfunc=format_fraction, fill_value="0/0").reset_index()

    # Final Save
    try:
        with pd.ExcelWriter(JUDGE_OUTPUT_FILE, engine='openpyxl') as writer:
            df_rank.to_excel(writer, sheet_name='Model Ranking', index=False)
            df_cat_emotion.to_excel(writer, sheet_name='Emotion Detection Details', index=False)
            df_eval.to_excel(writer, sheet_name='Evaluation Details', index=False)
            
            # Auto-width columns
            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    try:
                        max_len = max(len(str(cell.value)) for cell in col)
                        width = min(max_len + 2, 50)
                        sheet.column_dimensions[col[0].column_letter].width = width
                    except: pass

        log(f"Full report successfully saved: {JUDGE_OUTPUT_FILE}", "✅")
    except Exception as e:
        log(f"Error saving final Excel report: {e}", "❌")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Sequential Evaluator")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 1 case)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1)"
    )

    args = parser.parse_args()

    main(
        debug_mode=args.debug,
        max_workers=args.max_workers
    )