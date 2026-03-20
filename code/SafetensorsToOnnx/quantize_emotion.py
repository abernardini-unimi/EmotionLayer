import torch
import librosa
import json
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn as nn
import os

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

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
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.all_tied_weights_keys = {}

        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH   = "./emotion/safetensors"
ONNX_PATH    = "wav2vec2_emotion.onnx"
QUANT_PATH   = "wav2vec2_emotion_int8.onnx"
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
    model_input=ONNX_PATH,
    model_output=QUANT_PATH,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],  # ← esclude Conv → niente ConvInteger
)
print(f"✅ Modello quantizzato salvato: {QUANT_PATH}")
print(f"   Dimensione: {os.path.getsize(QUANT_PATH) / 1024**2:.1f} MB")
print(f"   Riduzione:  {100*(1 - os.path.getsize(QUANT_PATH)/os.path.getsize(ONNX_PATH)):.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENZA SUL FILE AUDIO
# ══════════════════════════════════════════════════════════════════════════════

# print(f"\n🎙️  Inferenza su: {AUDIO_FILE}")
# speech, sr = librosa.load(AUDIO_FILE, sr=16000)

# inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="np", padding=True)
# input_values   = inputs["input_values"].astype(np.float32)
# attention_mask = inputs["attention_mask"].astype(np.int64)

# # Carica sessione ONNX quantizzata
# sess_options = onnxruntime.SessionOptions()
# sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# session = onnxruntime.InferenceSession(QUANT_PATH, sess_options)

# ort_inputs = {
#     "input_values":   input_values,
#     "attention_mask": attention_mask,
# }

# predictions = session.run(["logits"], ort_inputs)[0][0]

# result = {
#     "arousal":   round(float(predictions[0]), 4),
#     "valence":   round(float(predictions[1]), 4),
#     "dominance": round(float(predictions[2]), 4),
# }

# print("\n📊 Risultato:")
# print(json.dumps(result, indent=4))