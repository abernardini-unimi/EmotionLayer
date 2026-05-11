# Affective Computing project 25/26

This repository contains the project developed for the **Affective Computing** and **Natural Interaction** courses.

The project provides a complete architecture for voice analysis and the evaluation of generative artificial intelligence. Specifically, it includes the code to run the *EmotionLayer* system locally (for extracting prosody and emotions from audio), test the empathy and response strategies of various LLMs, and consult the detailed benchmark results.

## 📁 Project Structure

The repository is organized into the following main components:

* **`paper.pdf`**: The complete research document illustrating the theoretical study, the architecture of the implemented system, and the analysis of the results obtained.
* **`code/`**: The directory containing all the project's source code. Inside, it is divided into three independent submodules. *Each folder contains its own dedicated README file with specific instructions for installation and execution.*

## 💻 Code Modules (`/code`)

Inside the `code` folder, you will find the following applications:

* **`EmotionLayer/`**
The integrated system for local voice analysis. It takes audio files as input and extracts prosodic values (Arousal, Valence, Dominance), predicts vocal emotions, transcribes the text, and calculates its sentiment.
* **`LlmEvaluation/`**
The testing framework based on an *LLM-as-a-Judge* approach. It is used to evaluate and compare different LLM models on their ability to recognize emotions and adopt the correct empathetic strategy.
**Note:** The detailed results of the tests already performed (Excel and JSON reports) can be found in the path `code/LlmEvaluation/file/output`.
* **`SafeTensorsToOnnx/`**
A utility tool containing the necessary scripts to convert model weights from the standard `.safetensors` format to the optimized `.ONNX` format, to speed up performance during inference.

## 🔗 Linked Repositories and Models

As an integral part of this research project, the voice analysis models used in *EmotionLayer* were trained from scratch or fine-tuned on Italian language datasets. Below are the links to the training code and the final open-source published models:

**Training Code (GitHub)**

* [Wav2Vec2.0-Italian-emotion](https://github.com/abernardini-unimi/Wav2Vec2.0-Italian-emotion): Source code used to train the model for vocal emotion classification.
* [Wav2Vec2.0-Italian-pad](https://github.com/abernardini-unimi/Wav2Vec2.0-Italian-pad): Source code used to train the model dedicated to predicting continuous prosodic values.

**Pre-trained Models (Hugging Face)**

* [wav2vec2-large-xlsr-53-italian-emotion-recognition](https://huggingface.co/abernardini-dev/wav2vec2-large-xlsr-53-italian-emotion-recognition): Final weights of the emotion prediction model, ready for inference use.
* [wav2vec2-large-xlsr-53-italian-pad](https://huggingface.co/abernardini-dev/wav2vec2-large-xlsr-53-italian-pad): Final weights of the model for prosody extraction, ready for inference use.

## 🚀 Getting Started

To start the various applications, please navigate to the folder of interest (e.g., `cd code/EmotionLayer`) and follow the setup instructions provided in the specific `README.md` file of that module. It will guide you through creating the virtual environment and installing the necessary dependencies.
