import time
import json
from google import genai
from google.genai import types # type:ignore
from config.config import GEMINI_API_KEY

# Inizializzazione Client
client = genai.Client(api_key=GEMINI_API_KEY)

def gemini_inference(model_name, system_prompt, user_input):
    """
    Esegue l'inferenza su Google Gemini.
    Restituisce un dizionario con risposta e metriche per il benchmark.
    """
    
    start_time = time.time()

    try:
        # Configurazione per forzare l'output JSON e gestire la temperatura
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json", # Fondamentale per la tesi
            temperature=0.3,
            max_output_tokens=2048
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config=config
        )
        
        latency = time.time() - start_time
        
        # Estrazione Usage Metadata (Google usa nomi leggermente diversi)
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        
        # Gestione del testo della risposta
        response_text = response.text

        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency
        }

    except Exception as e:
        print(f"Error in Gemini inference: {e}")
        return {
            "response": json.dumps({"error": str(e)}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": time.time() - start_time
        }

if __name__ == "__main__":
    # Test rapido
    model = "gemini-2.5-flash-lite" # O il modello che vuoi testare
    sys_prompt = "Sei un assistente che risponde solo in JSON."
    query = json.dumps({"test": "Ciao, chi sei?"})
    
    result = gemini_inference(model, sys_prompt, query)
    print(json.dumps(result, indent=2))