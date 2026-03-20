import torch
import librosa
import json
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn as nn
import os

# ══════════════════════════════════════════════════════════════════════════════
# DEFINIZIONE MODELLO (identica al tuo script originale)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            return torch.sum(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        raise Exception("Pooling mode not supported")

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        preds = torch.sigmoid(logits)
        return preds  # shape: [batch, 3]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH   = "pad/safetensor"
ONNX_PATH    = "wav2vec2_prosody.onnx"
QUANT_PATH   = "wav2vec2_prosody_int8.onnx"
AUDIO_FILE   = "registrazione_29.wav"

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARICA MODELLO PYTORCH
# ══════════════════════════════════════════════════════════════════════════════

print("📦 Caricamento modello PyTorch...")
config = AutoConfig.from_pretrained(MODEL_PATH)
setattr(config, 'pooling_mode', 'mean')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_PATH, config=config)
model.eval()

# ══════════════════════════════════════════════════════════════════════════════
# 2. CREA INPUT DUMMY PER EXPORT
# ══════════════════════════════════════════════════════════════════════════════

print("🔧 Creazione input dummy per export ONNX...")
dummy_audio = torch.zeros(1, 16000)  # 1 secondo di silenzio
dummy_mask  = torch.ones(1, 16000, dtype=torch.long)

# ══════════════════════════════════════════════════════════════════════════════
# 3. ESPORTA IN ONNX
# ══════════════════════════════════════════════════════════════════════════════

# ── 3. ESPORTA IN ONNX (vecchio exporter, stabile) ────────────────────────────
print("📤 Esportazione in ONNX...")

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_audio, dummy_mask),
        ONNX_PATH,
        input_names=["input_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_values":   {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits":         {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,   # ← forza il vecchio exporter TorchScript
    )
    
# Verifica
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print(f"✅ ONNX salvato e verificato: {ONNX_PATH}")
print(f"   Dimensione: {os.path.getsize(ONNX_PATH) / 1024**2:.1f} MB")

# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTIZZAZIONE INT8 DINAMICA
# ══════════════════════════════════════════════════════════════════════════════

print("⚙️  Quantizzazione INT8 dinamica...")
quantize_dynamic(
    model_input="wav2vec2_prosody.onnx",
    model_output="wav2vec2_prosody_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],  # ← esclude Conv → niente ConvInteger
)
print(f"✅ Modello quantizzato salvato: {QUANT_PATH}")
print(f"   Dimensione: {os.path.getsize(QUANT_PATH) / 1024**2:.1f} MB")
print(f"   Riduzione:  {100*(1 - os.path.getsize(QUANT_PATH)/os.path.getsize(ONNX_PATH)):.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENZA SUL FILE AUDIO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n🎙️  Inferenza su: {AUDIO_FILE}")
speech, sr = librosa.load(AUDIO_FILE, sr=16000)

inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="np", padding=True)
input_values   = inputs["input_values"].astype(np.float32)
attention_mask = inputs["attention_mask"].astype(np.int64)

# Carica sessione ONNX quantizzata
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(QUANT_PATH, sess_options)

ort_inputs = {
    "input_values":   input_values,
    "attention_mask": attention_mask,
}

predictions = session.run(["logits"], ort_inputs)[0][0]

result = {
    "arousal":   round(float(predictions[0]), 4),
    "valence":   round(float(predictions[1]), 4),
    "dominance": round(float(predictions[2]), 4),
}

print("\n📊 Risultato:")
print(json.dumps(result, indent=4))