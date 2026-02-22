import time
import json
from openai import OpenAI
from config.config import OPENAI_API_KEY

# Inizializzazione Client
client = OpenAI(api_key=OPENAI_API_KEY)

def openai_inference(model_name, system_prompt, user_input):
    """
    Esegue l'inferenza usando l'API ufficiale di OpenAI.
    Compatibile con il benchmark del main.py.
    """
    
    # Preparazione messaggi
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_input})

    start_time = time.time()

    temperature = 0.3
    if model_name == 'gpt-5-mini-2025-08-07':
        temperature=1

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature, # Basso per consistenza nel JSON
            response_format={"type": "json_object"}, # Forza l'output JSON 
        )

        latency = time.time() - start_time
        
        # Estrazione dati
        response_text = response.choices[0].message.content
        usage = response.usage
        
        return {
            "response": response_text,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "latency": latency
        }

    except Exception as e:
        print(f"Error in OpenAI inference: {e}")
        # Ritorna struttura vuota in caso di errore per non rompere il loop
        return {
            "response": json.dumps({"error": str(e)}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": time.time() - start_time
        }

# --- FUNZIONE STREAM (Opzionale, solo per debug visivo) ---
def openai_stream_response(query, model_id):
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": query}],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

# --- MAIN DI TEST ---
def main():
    query = json.dumps({"test": "ciao"}) # Simuliamo un input JSON
    model_id = "gpt-4o"
    system_prompt = "Sei un assistente che risponde solo in JSON."
    result = openai_inference(model_id, system_prompt, query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()