import os
import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        # Qui userà num_labels=3 dal tuo nuovo config
        self.out_proj = nn.Linear(config.hidden_size, len(config.id2label))

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
        self.config = config
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
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
        return torch.mean(hidden_states, dim=1)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values, attention_mask=None, return_dict=False)
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        # Usiamo Sigmoid perché arousal, valence, dominance sono valori indipendenti [0, 1]
        return torch.sigmoid(logits)

if __name__ == "__main__":
    percorso_modello = "./model"
    config = AutoConfig.from_pretrained(percorso_modello)
    modello = Wav2Vec2ForSpeechClassification.from_pretrained(percorso_modello, config=config)
    modello.eval()

    dummy_input = torch.randn(1, 16000)
    os.makedirs("onnx_output_nuovo", exist_ok=True)
    
    torch.onnx.export(
        modello, 
        dummy_input, 
        "onnx_output_nuovo/model.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_values'], 
        output_names=['logits'],
        dynamic_axes={'input_values': {0: 'batch_size', 1: 'sequence_length'}, 'logits': {0: 'batch_size'}}
    )
    print("Conversione VAD completata!")