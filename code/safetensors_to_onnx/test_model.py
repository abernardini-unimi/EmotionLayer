import numpy as np
import librosa
import onnxruntime as ort
from transformers import Wav2Vec2FeatureExtractor

# 1. Load the processor and the ONNX model
model_path = "onnx_output/model.onnx"
# Use the original folder path for the processor
processor = Wav2Vec2FeatureExtractor.from_pretrained("./model")
session = ort.InferenceSession(model_path)

# 2. Load and prepare the audio (must be 16kHz)
audio_path = "<your_audio_path>" 
speech, _ = librosa.load(audio_path, sr=16000)
inputs = processor(speech, sampling_rate=16000, return_tensors="np")

# 3. Run inference
# The input name is usually 'input_values' for Wav2Vec2
onnx_inputs = {session.get_inputs()[0].name: inputs.input_values}
logits = session.run(None, onnx_inputs)[0]

# 4. Decode the result
predicted_id = np.argmax(logits, axis=-1)[0]

# Manual mapping based on your config.json
emotions = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutrality", 5: "sadness", 6: "surprise"}
print(f"Detected emotion: {emotions[predicted_id]}")