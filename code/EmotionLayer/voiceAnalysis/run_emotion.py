from pathlib import Path
import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf 
from transformers import AutoConfig, Wav2Vec2FeatureExtractor 

from voiceAnalysis.src.models import Wav2Vec2ForSpeechClassification
# from src.models import Wav2Vec2ForSpeechClassification # Change ref if you are in the folder

model_path = 'voiceAnalysis/model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Prosody model from '{model_path}' su {device}...")

config = AutoConfig.from_pretrained(model_path)
setattr(config, 'pooling_mode', 'mean')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path, config=config)
model.to(device)
model.eval()
    

def speech_file_to_array_fn(audio_path, target_sampling_rate=16000):
    speech_array, src_sampling_rate = sf.read(audio_path)
    
    # From Numpy to Tensor Float
    speech_array = torch.from_numpy(speech_array).float()
    
    if speech_array.ndim == 1:
        speech_array = speech_array.unsqueeze(0) # (1, Time)
    else:
        speech_array = speech_array.t() # (Channels, Time)
        
    # If stereo, mean to mono
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    if src_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(src_sampling_rate, target_sampling_rate)
        speech_array = resampler(speech_array)
    
    speech = speech_array.squeeze().numpy()
    return speech


def get_emotion_info(audio_path):
    speech = speech_file_to_array_fn(audio_path)
    
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    outputs = []
    for i, score in enumerate(scores):
        outputs.append({
            "Emotion": model.config.id2label[i],
            "Score": round(float(score), 4) 
        })
        
    outputs.sort(key=lambda x: x["Score"], reverse=True)
    return outputs


def main():
    audio_path = "<your_file_path>"

    if Path(audio_path).exists():
        try:
            start_time = time.time()
            results = get_emotion_info(audio_path)
            end_time = time.time()
            
            json_output = json.dumps(results, indent=4)
            print(json_output)
            print(f'/nExecution Time : {end_time - start_time} s')
            
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
    else:
        print(json.dumps({"status": "error", "message": "File non trovato"}))
    
if __name__ == "__main__":
    main()