import numpy as np
import librosa
import onnxruntime
import json
import sys
import time
from transformers import Wav2Vec2FeatureExtractor

MODEL_PATH   = "pad/safetensor"   # per il feature extractor
QUANT_PATH   = "pad/onnx_quantize/wav2vec2_prosody_int8.onnx"
AUDIO_FILE   = "test6.wav"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(QUANT_PATH, sess_options)

def get_prosody_info(audio_path: str) -> dict:
    print(f"🎙️  Analisi file: {audio_path}")

    # Carica audio a 16kHz mono
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(speech) / 16000
    print(f"   Durata: {duration:.2f} secondi")

    # Feature extraction
    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="np",
        padding=True
    )

    ort_inputs = {
        "input_values":   inputs["input_values"].astype(np.float32),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Inferenza ONNX
    t_start = time.perf_counter()
    predictions = session.run(["logits"], ort_inputs)[0][0]
    t_end = time.perf_counter()

    inference_ms = (t_end - t_start) * 1000
    rtf = (t_end - t_start) / duration  # Real-Time Factor

    print(f"   ⏱️  Inferenza: {inference_ms:.1f} ms  (RTF: {rtf:.3f})")

    return {
        "arousal":   round(float(predictions[0]), 4),
        "valence":   round(float(predictions[1]), 4),
        "dominance": round(float(predictions[2]), 4),
    }

if __name__ == "__main__":

    result = get_prosody_info(AUDIO_FILE)

    print("\n📊 Risultato:")
    print(json.dumps(result, indent=4))
    print()
    print(f"   Arousal   (energia):    {result['arousal']:.4f}  {'🔴 alta' if result['arousal'] > 0.6 else '🟡 media' if result['arousal'] > 0.4 else '🟢 bassa'}")
    print(f"   Valence   (positività): {result['valence']:.4f}  {'😊 positiva' if result['valence'] > 0.6 else '😐 neutra' if result['valence'] > 0.4 else '😟 negativa'}")
    print(f"   Dominance (controllo):  {result['dominance']:.4f}  {'💪 alta' if result['dominance'] > 0.6 else '🤝 media' if result['dominance'] > 0.4 else '🙇 bassa'}")