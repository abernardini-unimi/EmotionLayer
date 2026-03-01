import librosa
import numpy as np
import onnxruntime as ort
import json
from transformers import AutoFeatureExtractor, AutoConfig

# Percorsi
percorso_modello_originale = "./onnx_emotion_model"
percorso_onnx = "onnx_emotion_model/model.onnx"
file_audio_test = "registrazione_29.wav"

# 1. Setup
estrattore = AutoFeatureExtractor.from_pretrained(percorso_modello_originale)
config = AutoConfig.from_pretrained(percorso_modello_originale)
id2label = config.id2label

# 2. Pre-elaborazione dell'audio
audio_array, sampling_rate = librosa.load(file_audio_test, sr=16000)
inputs = estrattore(audio_array, sampling_rate=sampling_rate, return_tensors="np")
input_values = inputs["input_values"]

# 3. Inferenza ONNX
sessione = ort.InferenceSession(percorso_onnx)
nome_input_onnx = sessione.get_inputs()[0].name
risultato_onnx = sessione.run(None, {nome_input_onnx: input_values})
logits = risultato_onnx[0][0] # Estraiamo i logit del primo (e unico) elemento del batch

# 4. Calcolo delle probabilità (Softmax in numpy)
esponenziali = np.exp(logits - np.max(logits)) # Sottrazione per stabilità numerica
probabilita = esponenziali / esponenziali.sum()

# 5. Creazione della lista delle emozioni formattata
risultati_emozioni = []
for i, prob in enumerate(probabilita):
    # Gestisce sia chiavi intere che stringhe nel dizionario
    nome_emozione = id2label.get(i, id2label.get(str(i))) 
    risultati_emozioni.append({
        "label": nome_emozione,
        "score": round(float(prob), 4) # Arrotondiamo a 4 decimali per pulizia
    })

# Ordiniamo la lista dalla probabilità più alta alla più bassa
risultati_emozioni = sorted(risultati_emozioni, key=lambda x: x["score"], reverse=True)

# Stampiamo il risultato in formato JSON formattato correttamente
print(json.dumps(risultati_emozioni, indent=4))