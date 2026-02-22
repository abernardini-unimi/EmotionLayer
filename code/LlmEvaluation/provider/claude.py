import time
import json
from anthropic import Anthropic
from config.config import ANTHROPIC_API_KEY

# Inizializzazione Client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def claude_inference(model_name, system_prompt, user_input):
    """
    Esegue l'inferenza su modelli Anthropic.
    Ritorna la risposta grezza (Raw JSON) senza wrapper aggiuntivi.
    """
    
    start_time = time.time()

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=0.3, 
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        latency = time.time() - start_time
        
        # Estrazione testo
        response_text = response.content[0].text
        
        # Estrazione metriche usage
        usage = response.usage
        
        return {
            "response": response_text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency": latency
        }

    except Exception as e:
        print(f"Error in Claude inference: {e}")
        return {
            "response": json.dumps({"error": str(e)}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": time.time() - start_time
        }

if __name__ == "__main__":
    prompt = "Sei un assistente che risponde solo in JSON."
    query = json.dumps({"domanda": "Ciao, chi sei?"})
    
    model = "claude-3-5-sonnet-20240620" 
    
    result = claude_inference(model, prompt, query)
    print(json.dumps(result, indent=2))