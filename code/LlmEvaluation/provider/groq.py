import time
import json
from groq import Groq # type: ignore
from config.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def groq_inference(model_name, system_prompt, user_input):
    """
    Esegue l'inferenza su Groq.
    Restituisce un dizionario con risposta e metriche per il benchmark.
    """

    case = """
    IMPORTANTE:
    1. Rispondi SOLO con un oggetto JSON valido.
    2. NON utilizzare blocchi di codice Markdown (niente ```json o ```).
    3. La tua risposta deve iniziare tassativamente con il carattere "{".
    """
    
    messages = [
        {"role": "system", "content": system_prompt if model_name != 'meta-llama/llama-guard-4-12b' else system_prompt + case},
        {"role": "user", "content": user_input}
    ]

    start_time = time.time()

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3, # Basso per stabilità JSON
            max_tokens=8192 if model_name != 'meta-llama/llama-guard-4-12b' else 1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}, # Forza output JSON
            stop=None
        )
        
        latency = time.time() - start_time
        
        # Estrazione dati
        response_text = completion.choices[0].message.content
        usage = completion.usage

        return {
            "response": response_text,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "latency": latency
        }

    except Exception as e:
        print(f"Error in Groq inference: {e}")
        return {
            "response": json.dumps({"error": str(e)}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": time.time() - start_time
        }

if __name__ == "__main__":
    # Test rapido
    prompt = "Sei un assistente che risponde solo in JSON."
    query = json.dumps({"domanda": "Ciao, chi sei?"})
    
    result = groq_inference("llama-3.1-8b-instant", prompt, query)
    print(json.dumps(result, indent=2))