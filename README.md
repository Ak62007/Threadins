# 🧥 Fashion Advisor Model for Indian College Students

A specialized LLM fine-tuned to provide fashion advice tailored specifically for Indian college students.

## 📋 Project Overview

This project fine-tunes a Mistral-7B model to create a fashion advisor that can answer clothing and style questions specifically for Indian college students. The model is trained on a dataset of fashion-related questions and answers, making it useful for providing contextually appropriate fashion guidance.

## ✨ Features

- **Domain-Specific Training**: Fine-tuned on fashion Q&A data specifically for Indian college students
- **Memory-Efficient Training**: Uses 4-bit quantization and LoRA for efficient training on consumer hardware
- **Conversational Interface**: Responds to fashion-related queries in a helpful, natural manner

## 🛠️ Technical Implementation

- **Base Model**: mistralai/Mistral-7B-Instruct-v0.3
- **Training Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- **Quantization**: 4-bit quantization with BitsAndBytes
- **Optimization**: Flash Attention 2 (when available) for speed improvements

## 🔧 Prerequisites

- Python 3.8+
- PyTorch
- 16GB+ RAM
- CUDA-capable GPU recommended (8GB+ VRAM)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fashion-advisor-model.git
cd fashion-advisor-model

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Required Dependencies

Create a `requirements.txt` file with the following dependencies:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
bitsandbytes>=0.40.0
peft>=0.4.0
accelerate>=0.20.0
python-docx>=0.8.11
```

## 🚀 Usage

### Training the Model

1. Place your fashion Q&A data in a Word document format (see data format below)
2. Update the file path in the `main()` function:

```python
file_path = "path/to/your/data.docx"  # Update with your data file path
```

3. Run the training script:

```bash
python train_fashion_advisor.py
```

The model will be saved to the directory specified in the `output_dir` parameter (default: `./fashion_advisor_model`).

### Using the Model

After training, you can use the model to generate fashion advice:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_path = "./fashion_advisor_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Create text generation pipeline
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=512, 
    do_sample=True, 
    temperature=0.7
)

# Example prompt
prompt = "Answer the question for an Indian college student's fashion doubt: What should I wear for a college fest?\nResponse:"

# Generate response
response = generator(prompt)
print(response[0]['generated_text'])
```

## 📊 Data Format

The training script expects a Word document (.docx) with Q&A pairs in the following format:

```
Q1) What should I wear to a college interview?
Professional attire such as formal pants or a skirt with a button-down shirt or blouse is recommended. Avoid overly casual clothing like t-shirts or ripped jeans.

Q2) How can I style my kurta for daily college wear?
You can pair a kurta with jeans for a fusion look. Add simple jewelry and comfortable footwear like juttis or sneakers.

...
```

## 🔄 Training Process

The training process consists of the following steps:

1. Data loading and preprocessing from Word document
2. Creation of training and validation datasets
3. Formatting prompts with appropriate context
4. Setting up the Mistral-7B model with 4-bit quantization
5. Applying LoRA adapters for parameter-efficient fine-tuning
6. Tokenizing the dataset
7. Training the model with optimized parameters
8. Saving the fine-tuned model
9. Testing the model with sample queries

## 🚧 Limitations

- The model's advice is limited to the training data provided
- The model may not be suitable for professional styling advice
- Performance depends on the quality and diversity of the training data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Insert your license information here]

## 🙏 Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base model
- [Hugging Face](https://huggingface.co/) for the Transformers library
