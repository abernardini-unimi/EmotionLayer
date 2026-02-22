import os
import time
import json
from openai import OpenAI # type: ignore
from config.config import HF_TOKEN

# Inizializzazione Client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

def hf_inference(model_name, system_prompt, user_input):
    """
    Esegue l'inferenza usando l'API OpenAI-compatible di Hugging Face.
    Restituisce un dizionario con risposta e metriche.
    """
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": user_input
    })

    start_time = time.time()
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3, # Bassa temperatura per output JSON stabile
            max_tokens=512,
            response_format={"type": "json_object"}, # Tenta di forzare il JSON
            stream=False
        )
        
        latency = time.time() - start_time

        # Estrazione token usage (Hugging Face di solito li restituisce)
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        response_text = completion.choices[0].message.content

        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency
        }

    except Exception as e:
        # Gestione errori API per evitare che il benchmark si blocchi
        return {
            "response": json.dumps({"error": str(e)}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": time.time() - start_time
        }

# --- FUNZIONE STREAM (Opzionale, utile per debug visivo) ---
def hf_stream_response(query, model_id): 
    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": query}
        ],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)

# --- MAIN DI TEST (Per provare solo questo file) ---
def main():
    model_id = "Qwen/Qwen2.5-72B-Instruct:featherless-ai" # Assicurati che il modello esista su HF Inference
    
    result = hf_inference(
        model_name=model_id, 
        system_prompt="[INST] Sei un utile assistente. Rispondi nella stessa lingua dell'utente [/INST]", 
        user_input="Ciao! chi sei?"
    )
    
    print("Risultato ottenuto:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()