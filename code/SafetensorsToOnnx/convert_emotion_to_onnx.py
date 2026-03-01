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

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("The pooling method hasn't been defined!")
        return outputs

    def forward(self, input_values, attention_mask=None, return_dict=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            return_dict=False, # Forziamo l'uscita in tupla per evitare oggetti custom
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        return logits


# --- 2. WRAPPER PER L'ESPORTAZIONE ONNX ---
# ONNX vuole in ingresso un modulo che sputi fuori SOLO il tensore finale
class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        # Chiamiamo il forward del tuo modello e restituiamo solo le logits
        return self.model(input_values)


# --- 3. CARICAMENTO ED ESPORTAZIONE ---
if __name__ == "__main__":
    print("Caricamento della configurazione e del modello...")
    
    config = AutoConfig.from_pretrained("./model")
    modello_originale = Wav2Vec2ForSpeechClassification.from_pretrained("./model", config=config)
    modello_originale.eval()

    # Avvolgiamo il modello nel wrapper per ONNX
    modello_onnx_ready = ONNXWrapper(modello_originale)

    # Input fittizio (es. batch 1, 16000 campioni)
    dummy_input = torch.randn(1, 16000)

    os.makedirs("onnx_output", exist_ok=True)
    percorso_onnx = "onnx_output/model.onnx"

    print("Inizio esportazione ONNX (opset 18)...")
    torch.onnx.export(
        modello_onnx_ready, 
        dummy_input, 
        percorso_onnx, 
        export_params=True,
        opset_version=18,    # <-- Aggiornato a 18 per evitare il crash
        do_constant_folding=True,
        input_names=['input_values'], 
        output_names=['logits'],
        dynamic_axes={
            'input_values': {0: 'batch_size', 1: 'sequence_length'}, 
            'logits': {0: 'batch_size'}
        }
    )
    print(f"\nEsportazione completata con successo in: {percorso_onnx}")