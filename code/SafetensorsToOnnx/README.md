# Safetensors to ONNX Conversion

This folder is **only** used to convert a model stored in **Safetensors** format into **ONNX** format using the CLI provided by **Hugging Face Optimum**.

## Requirements

Make sure you have:

* Python ≥ 3.8
* `optimum` with ONNX Runtime support

Install the required dependencies with:

```bash
pip install "optimum[onnxruntime]"
```

## Expected Folder Structure

```
.
├── model/              # Folder containing the Safetensors model
└── onnx_output/        # (Will be created) Output folder for the ONNX model
```

The `model/` directory should contain:

* `config.json`
* `model.safetensors`
* Any additional required files (e.g., tokenizer, processor, etc.)

## Conversion Command

Run the following command from the root of this folder:

```bash
optimum-cli export onnx \
  --model ./model \
  --task audio-classification \
  onnx_output/
```

### Main Parameters

* `--model ./model` → Path to the Safetensors model directory
* `--task audio-classification` → Model task (change if needed)
* `onnx_output/` → Destination directory for the exported ONNX model

## Output

After the process completes, the converted ONNX model will be available inside the `onnx_output/` directory.
