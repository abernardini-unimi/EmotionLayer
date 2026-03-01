from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Definisci le cartelle di input e output
cartella_input = "pad_model"
cartella_output = "quant_pad_model"

print("Caricamento del modello ONNX tramite Optimum...")

# 1. Inizializza il quantizzatore puntando alla cartella contenente model.onnx e config.json
quantizer = ORTQuantizer.from_pretrained(cartella_input, file_name="model.onnx")

# 2. Configura la quantizzazione dinamica standard (ideale per CPU)
# avx2 è il set di istruzioni standard per ottimizzare su CPU
dqconfig = AutoQuantizationConfig.avx2(is_static=False)

print("Avvio la quantizzazione dinamica...")

# 3. Applica la quantizzazione e salva nella nuova cartella
quantizer.quantize(
    save_dir=cartella_output,
    quantization_config=dqconfig,
)

print(f"Successo! Il modello quantizzato e il nuovo config sono salvati in: {cartella_output}")