import os
import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig
from onnxruntime.quantization import quantize_dynamic, QuantType

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
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
        return torch.sigmoid(logits)

if __name__ == "__main__":
    percorso_modello = "./model"
    config = AutoConfig.from_pretrained(percorso_modello)
    modello = Wav2Vec2ForSpeechClassification.from_pretrained(percorso_modello, config=config)
    modello.eval()

    dummy_input = torch.randn(1, 16000)
    cartella_output = "onnx_output_nuovo"
    os.makedirs(cartella_output, exist_ok=True)
    
    percorso_onnx = f"{cartella_output}/model.onnx"
    percorso_quantizzato = f"{cartella_output}/model_quantized.onnx"
    
    print("1. Eseguo il TorchScript Tracing per disabilitare Dynamo...")
    # Questo passaggio è la chiave di tutto: congela il modello in formato classico
    with torch.no_grad():
        traced_model = torch.jit.trace(modello, dummy_input, strict=False)
    
    print("2. Avvio l'esportazione ONNX (Legacy Mode)...")
    torch.onnx.export(
        traced_model,  # Passiamo il modello TRACCIATO, non quello normale
        dummy_input, 
        percorso_onnx,
        export_params=True,
        opset_version=14, # Ripristiniamo il 14 che è perfetto per Wav2Vec2
        do_constant_folding=True,
        input_names=['input_values'], 
        output_names=['logits'],
        dynamic_axes={
            'input_values': {1: 'sequence_length'} 
        }
    )
    print("Esportazione ONNX completata con successo!")

    print("3. Avvio la quantizzazione dinamica del file esportato...")
    quantize_dynamic(
        model_input=percorso_onnx,
        model_output=percorso_quantizzato,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True
    )
    print(f"BINGO! Il modello è quantizzato e pronto in: {percorso_quantizzato}")