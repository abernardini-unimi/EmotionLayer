import torch #type: ignore
import librosa #type: ignore
import sys
import json

from prosodyAnalysis.src.models import Wav2Vec2ForSpeechClassification 
# from src.models import Wav2Vec2ForSpeechClassification # Change ref if you are in the folder
from transformers import Wav2Vec2FeatureExtractor, AutoConfig #type: ignore

model_path = 'prosodyAnalysis/model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Prosody model from '{model_path}' su {device}...")

config = AutoConfig.from_pretrained(model_path)
setattr(config, 'pooling_mode', 'mean')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path, config=config)
model.to(device)
model.eval()


def get_prosody_info(audio_path):
    print(f"Analysis file: {audio_path}")
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.cpu().numpy()[0]

    print(predictions)

    arousal = float(f"{predictions[0]:.4f}")
    valence = float(f"{predictions[1]:.4f}")
    dominance = float(f"{predictions[2]:.4f}")

    return {
        "arousal": arousal, 
        "valence": valence, 
        "dominance": dominance
    }

if __name__ == "__main__":
    file_audio = "registrazione_29.wav"
    if len(sys.argv) > 1:
        file_audio = sys.argv[1]

    result = get_prosody_info(file_audio)
    
    print("Result:")
    print(json.dumps(result, indent=4))