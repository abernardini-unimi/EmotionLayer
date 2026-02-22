import json
import os
import pandas as pd

def load_text_file(filename):
    """Load the content of a text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: File '{filename}' not found.")
        exit(1)

def load_json_from_file(filename):
    """Load the test dataset from a JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ Dataset loaded successfully: {len(data)} test cases found in '{filename}'.")
            return data
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file '{filename}' not found.")
        print(f"Please check that the path is correct: {os.path.abspath(filename)}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: File '{filename}' is not a valid JSON.")
        print(f"Error details: {e}")
        exit(1)

def extract_real_json(raw_input):
    """
    Robust function to extract JSON from messy strings.
    Handles: Markdown, JSON inside strings, Wrappers like {"response": ...}
    """
    if not isinstance(raw_input, str):
        return {}

    candidate = raw_input.strip()

    # 1. Remove Markdown blocks (```json ... ```)
    if "```" in candidate:
        try:
            if "```json" in candidate:
                candidate = candidate.split("```json")[1].split("```")[0]
            else:
                candidate = candidate.split("```")[1]
        except IndexError:
            pass
    
    candidate = candidate.strip()

    # 2. Iterative parsing (handles JSON inside JSON strings)
    try:
        parsed = json.loads(candidate)
        
        # If result is still a string, try parsing again
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except:
                pass 
        
        # 3. Handle wrapper {"response": ...} typical of some APIs
        if isinstance(parsed, dict) and "response" in parsed:
            inner = parsed["response"]
            if isinstance(inner, str):
                try:
                    return extract_real_json(inner) # recursion
                except:
                    return inner
            elif isinstance(inner, dict):
                return inner
            
        return parsed
            
    except json.JSONDecodeError:
        return {}

def generate_report(results_data, output_filename):
    """
    Generate an Excel file with advanced statistics and readable costs.
    """
    print(f"📊 Generating Excel Report: {output_filename}...")
    flattened_rows = []

    for case in results_data:
        input_data = case.get('original_input', {})
        prosody = input_data.get('prosody', {})
        
        base_row = {
            'Test Case ID': case.get('test_case_id'),
            'Emotion Detected': case.get('emotion_detected'),
            'Target Strategy': case.get('target_strategy'),
            'Input Text': input_data.get('transcription'),
            'Vocal Sentiment': json.dumps(input_data.get('vocal_sentiment'), ensure_ascii=False),
            'Arousal': prosody.get('arousal'),
            'Dominance': prosody.get('dominance'),
            'Valence': prosody.get('valence')
        }

        for model_name, res in case.get('benchmark_results', {}).items():
            row = base_row.copy()
            row['Model'] = model_name
            row['Provider'] = res.get('provider')
            
            metrics = res.get('metrics', {})
            row['Latency (s)'] = metrics.get('latency_s', 0)
            row['Cost ($)'] = metrics.get('cost_usd', 0)
            row['Input Tokens'] = metrics.get('input_tokens', 0)
            row['Output Tokens'] = metrics.get('output_tokens', 0)
            
            final_json = extract_real_json(res.get('response_raw', ''))
            
            if isinstance(final_json, dict) and ('reasoning' in final_json or 'selected_strategy' in final_json):
                row['Valid JSON'] = 1
                row['Reasoning'] = final_json.get('reasoning')
                row['Emotion Detected'] = final_json.get('emotion_detected')
                row['Selected Strategy'] = final_json.get('selected_strategy')
                row['Response Text'] = final_json.get('response_text')
                tts = final_json.get('tts_config', {})
                row['Speed'] = tts.get('speed')
                row['Tone'] = tts.get('tone')
                row['Pitch'] = tts.get('pitch')
            else:
                row['Valid JSON'] = 0
                row['Reasoning'] = "PARSING FAILED"
                row['Response Text'] = str(res.get('response_raw', ''))

            flattened_rows.append(row)

    if not flattened_rows:
        print("⚠️ No data to save.")
        return

    df_detail = pd.DataFrame(flattened_rows)

    # --- CALCULATE AGGREGATED STATISTICS ---
    print("📈 Calculating aggregated statistics...")
    df_stats = df_detail.groupby(['Provider', 'Model']).agg({
        'Latency (s)': 'mean',
        'Cost ($)': 'mean',
        'Input Tokens': 'mean',
        'Output Tokens': 'mean',
        'Valid JSON': 'mean',
        'Test Case ID': 'count'
    }).reset_index()

    # Rename columns
    df_stats.rename(columns={
        'Latency (s)': 'Avg Latency (s)',
        'Cost ($)': 'Avg Cost ($)',
        'Input Tokens': 'Avg Input Tokens',
        'Output Tokens': 'Avg Output Tokens',
        'Valid JSON': 'Success Rate (%)',
        'Test Case ID': 'Total Tests'
    }, inplace=True)

    # Formatting
    df_stats['Success Rate (%)'] = df_stats['Success Rate (%)'] * 100
    
    # Smart rounding
    df_stats = df_stats.round({
        'Avg Latency (s)': 3,
        'Avg Cost ($)': 9,          
        'Avg Input Tokens': 1,
        'Avg Output Tokens': 1,
        'Success Rate (%)': 1
    })
    
    # Sort by latency
    df_stats = df_stats.sort_values(by='Avg Latency (s)', ascending=True)

    # Reorder columns to keep costs close
    cols = ['Provider', 'Model', 'Avg Latency (s)', 'Success Rate (%)', 'Avg Cost ($)', 
            'Avg Input Tokens', 'Avg Output Tokens', 'Total Tests']
    df_stats = df_stats[cols]

    # --- SAVE TO EXCEL ---
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            df_detail.to_excel(writer, sheet_name='Details', index=False)
            
            # Auto-adjust column width
            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    try:
                        max_len = max(len(str(cell.value)) for cell in col)
                        width = min(max_len + 2, 60)
                        sheet.column_dimensions[col[0].column_letter].width = width
                    except: pass
        print(f"✅ Excel saved successfully: {output_filename}")
        
    except Exception as e:
        print(f"❌ Excel saving error: {e}")