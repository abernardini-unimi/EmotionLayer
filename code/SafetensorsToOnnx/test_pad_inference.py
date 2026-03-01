import librosa
import numpy as np
import onnxruntime as ort
import json
from transformers import AutoFeatureExtractor, AutoConfig

percorso_modello = "./onnx_pad_model"
percorso_onnx = "onnx_pad_model/model.onnx"
file_audio = "registrazione_29.wav"

# 1. Setup
estrattore = AutoFeatureExtractor.from_pretrained(percorso_modello)
config = AutoConfig.from_pretrained(percorso_modello)
id2label = config.id2label # Questo ora contiene arousal, valence, dominance

# 2. Audio
audio, _ = librosa.load(file_audio, sr=16000)
inputs = estrattore(audio, sampling_rate=16000, return_tensors="np")

# 3. ONNX
sessione = ort.InferenceSession(percorso_onnx)
output_onnx = sessione.run(None, {"input_values": inputs["input_values"]})
valori = output_onnx[0][0]

# 4. Mappatura dinamica basata sul JSON
risultato_finale = {}
for i in range(len(valori)):
    label = id2label.get(i, id2label.get(str(i)))
    risultato_finale[label] = round(float(valori[i]), 4)

print(json.dumps(risultato_finale, indent=4))