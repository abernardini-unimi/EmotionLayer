import onnxruntime
import librosa
import numpy as np
import json
from transformers import Wav2Vec2FeatureExtractor, AutoConfig

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = "emotion/safetensors"   # per il feature extractor e config
QUANT_PATH   = "emotion/onnx_quantize/wav2vec2_emotion_int8.onnx"
AUDIO_FILE   = "test6.wav"

def softmax(x):
    """Calcola la softmax per trasformare i logits in probabilità (0-1)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main():
    # 1. Caricamento dell'audio
    print(f"🎙️ Caricamento file audio: {AUDIO_FILE}...")
    try:
        speech, sr = librosa.load(AUDIO_FILE, sr=16000)
    except FileNotFoundError:
        print(f"❌ Errore: Il file {AUDIO_FILE} non è stato trovato.")
        return

    # 2. Inizializzazione Feature Extractor e Config
    print("⚙️ Caricamento feature extractor e configurazione...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="np", padding=True)
    input_values   = inputs["input_values"].astype(np.float32)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # 3. Setup di ONNX Runtime
    print(f"🚀 Avvio inferenza sul modello quantizzato ({QUANT_PATH})...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(QUANT_PATH, sess_options)

    # 4. Inferenza
    ort_inputs = {
        "input_values":   input_values,
        "attention_mask": attention_mask,
    }

    predictions = session.run(["logits"], ort_inputs)[0][0]

    # Trasformiamo i valori grezzi in probabilità
    probabilities = softmax(predictions)

    # 5. Formattazione dei risultati tramite id2label
    id2label = config.id2label
    result = {}
    
    for i, prob in enumerate(probabilities):
        # Cerca la chiave sia come stringa che come intero per evitare KeyError
        label_name = id2label.get(str(i), id2label.get(i, f"LABEL_{i}"))
        result[label_name] = round(float(prob), 4)

    # Ordiniamo il dizionario in base al valore (probabilità decrescente)
    result_sorted = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    print("\n📊 Risultato dell'analisi (Probabilità):")
    print(json.dumps(result_sorted, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()