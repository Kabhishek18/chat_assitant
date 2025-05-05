# Hybrid AI Model Development & Inference System
## Leveraging Hugging Face for Training and Ollama for Inference on Apple Silicon

![Apple Silicon + AI Banner](https://source.unsplash.com/random/1200x400/?ai,chip)

## Overview

This project implements a hybrid approach to AI model development and deployment, specifically optimized for Apple Silicon Macs (M-series chips). It uses Hugging Face for model fine-tuning and development, and Ollama for optimized local inference.

### Why This Hybrid Approach?

This approach allows you to:
- Fine-tune models using Hugging Face's extensive tooling and ecosystem
- Deploy models locally with Ollama's performance-optimized inference
- Maintain complete data privacy with on-premise deployment
- Achieve fast inference speeds with lower latency
- Work efficiently within Apple Silicon hardware constraints

## 🚀 System Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3 series), 16GB+ RAM recommended
- **Operating System**: macOS Sonoma 14.0+ (earlier versions may work but are not fully tested)
- **Storage**: 20GB+ free space for models and development environment
- **Python**: Python 3.10+ (arm64 build)

## 📋 Getting Started

### 1. Environment Setup

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python cmake

# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio

# Install Hugging Face libraries
pip install transformers datasets accelerate huggingface_hub
```

### 2. Install Ollama

Download and install Ollama from the official website: [https://ollama.com/download/mac](https://ollama.com/download/mac)

Or install via command line:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Verify MPS Support

Run the following Python script to verify Metal support:

```python
import torch
print(f"Is MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"Is built with MPS: {torch.backends.mps.is_built()}")
```

## 🔄 Workflow

### Development Phase (Hugging Face)

1. **Select Base Model**
   - Choose smaller models (7B-13B parameters) that work efficiently on M3 Pro
   - Evaluate pre-trained models from Hugging Face Hub

2. **Data Preparation**
   - Prepare and preprocess your training data
   - Create validation datasets for evaluation

3. **Fine-Tuning**
   - Use parameter-efficient techniques like LoRA or QLoRA
   - Implement the following training script template:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Configure for Metal Performance Shaders
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
model_name = "your-base-model" # e.g., "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to MPS device
model = model.to(device)

# Load and prepare dataset
dataset = load_dataset("your-dataset")
# ... preprocess dataset ...

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_strategy="epoch",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```

### Deployment Phase (Ollama)

1. **Convert Model to GGUF**
   - Use llama.cpp or similar tools to convert and quantize your model

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the conversion tools
make

# Convert model (example for Hugging Face format to GGUF)
python convert.py /path/to/fine-tuned-model --outfile models/your-model-q4_0.gguf --outtype q4_0
```

2. **Create Modelfile for Ollama**

```
FROM ./models/your-model-q4_0.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
SYSTEM """
Your custom system prompt goes here.
"""
```

3. **Load Model into Ollama**

```bash
# Create a model in Ollama using your Modelfile
ollama create your-model-name -f /path/to/Modelfile

# Test your model
ollama run your-model-name "Your test prompt here"
```

## 📊 Performance Optimization

### Memory Management

- Adjust Ollama memory settings in configuration:

```bash
# Set memory limit for Ollama
OLLAMA_HOST=0.0.0.0 OLLAMA_MODELS=/path/to/models OLLAMA_RAM=12000000000 ollama serve
```

- Use gradient accumulation for training larger models:

```python
# In training_args
gradient_accumulation_steps=8,  # Increase for larger effective batch size
```

### Quantization Recommendations

For M3 Pro with 18GB RAM:
- Try Q4_K_M first for a good balance of quality and performance
- Use Q6_K for higher quality if memory permits
- Fall back to Q4_0 for largest models if necessary

## 🔌 API Integration

Create a simple API server to interface with your Ollama models:

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'your-default-model')
    
    response = requests.post('http://localhost:11434/api/generate', 
                           json={
                               'model': model,
                               'prompt': prompt
                           })
    
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📝 Maintenance Guidelines

### Model Updates

1. Regularly check for new base models on Hugging Face
2. Implement a versioning system for your fine-tuned models
3. Create an update schedule that includes:
   - Re-evaluation on benchmark datasets
   - Fine-tuning with new data
   - Conversion to latest GGUF format

### Performance Monitoring

1. Monitor CPU, GPU, and memory usage during inference:

```bash
# Using built-in macOS tools
sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 60
```

2. Track inference times and thermal performance
3. Adjust batch sizes and quantization based on performance metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Try smaller models
   - Use more aggressive quantization
   - Close other applications

2. **Slow Inference**
   - Check for thermal throttling
   - Verify MPS is being used
   - Try different quantization levels
   - Optimize prompt length

3. **Model Loading Failures**
   - Verify GGUF conversion was successful
   - Check Modelfile syntax
   - Ensure sufficient disk space

## 🔗 Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Apple Metal Developer Documentation](https://developer.apple.com/documentation/metal)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for their transformers library and model hub
- Ollama team for optimized local inference
- llama.cpp project for efficient model conversion and quantization
- Apple for Metal Performance Shaders support

---

*Last Updated: May 2025*
