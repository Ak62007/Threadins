import re
import os
import torch
import pandas as pd
import numpy as np
from docx import Document
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# --------------------------------------------------
# 1. Data Loading and Preprocessing
# --------------------------------------------------

def load_data_from_docx(file_path):
    """Load and parse Q&A pairs from Word document."""
    print(f"Loading data from {file_path}...")
    doc = Document(file_path)
    
    # Extract paragraphs
    raw_data = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    
    # Initialize lists to store questions and answers
    questions = []
    answers = []
    
    # Regex pattern to identify questions
    question_pattern = re.compile(r"^Q\d+\)")
    
    # Temporary variables
    current_question = None
    current_answer = []
    
    # Process the extracted text
    for line in raw_data:
        if question_pattern.match(line):  # Line is a question
            if current_question and current_answer:
                # Save the previous question-answer pair
                answers.append(" ".join(current_answer).strip())
                current_answer = []
    
            questions.append(line)
            current_question = line
        else:
            current_answer.append(line)
    
    # Add the last question-answer pair
    if current_question and current_answer:
        answers.append(" ".join(current_answer).strip())
    
    # Ensure questions and answers are aligned
    if len(questions) != len(answers):
        print(f"Warning: Mismatched questions ({len(questions)}) and answers ({len(answers)}).")
    else:
        print(f"Successfully processed {len(questions)} question-answer pairs.")
    
    # Create a DataFrame
    data = {"question": questions, "answer": answers}
    df = pd.DataFrame(data)
    
    return df

def clean_and_prepare_data(df):
    """Clean question format and prepare the data."""
    # Clean question format (remove Q1), Q2), etc.)
    df["question"] = df["question"].apply(
        lambda q: re.sub(r"^Q\d+\)\s*", "", q).strip()
    )
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Save as CSV
    train_csv_path = "fashion_qa_train.csv"
    val_csv_path = "fashion_qa_val.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Training dataset saved as {train_csv_path} with {len(train_df)} records.")
    print(f"Validation dataset saved as {val_csv_path} with {len(val_df)} records.")
    
    return train_csv_path, val_csv_path

# --------------------------------------------------
# 2. Format and Load Dataset
# --------------------------------------------------

def format_and_load_dataset(train_csv_path, val_csv_path):
    """Format prompts and load the dataset."""
    # Load dataset from CSVs
    data_files = {"train": train_csv_path, "validation": val_csv_path}
    dataset = load_dataset("csv", data_files=data_files)
    
    # Add the prompt formatting step to the dataset
    def add_prompt(example):
        example["prompt"] = f"Answer the question for an Indian college student's fashion doubt: {example['question']}\nResponse: {example['answer']}"
        return example
    
    # Map the formatting over both splits
    dataset = dataset.map(add_prompt)
    
    # Print sample for verification
    print("\nSample formatted prompt:")
    print(dataset["train"][0]["prompt"])
    
    return dataset

# --------------------------------------------------
# 3. Model Setup
# --------------------------------------------------

def setup_model_and_tokenizer(checkpoint="mistralai/Mistral-7B-Instruct-v0.3", use_flash_attention=True):
    """Set up the model and tokenizer with appropriate configuration."""
    print(f"\nLoading model and tokenizer from {checkpoint}...")
    
    # Set up 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Reduce memory usage
        bnb_4bit_use_double_quant=True,  # More compression
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading arguments
    model_args = {
        "quantization_config": bnb_config,
        "device_map": "auto",  # Auto distributes between CPU and GPU
        "trust_remote_code": True
    }
    
    # Add flash attention if requested
    if use_flash_attention:
        try:
            model_args["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2 for faster inference")
        except Exception as e:
            print(f"Flash Attention not available: {e}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_args)
    
    # Print memory allocation info
    print("Model device map:")
    print(model.hf_device_map)
    
    return model, tokenizer

def prepare_model_for_training(model, checkpoint):
    """Prepare the model for LoRA fine-tuning."""
    print("\nPreparing model for training...")
    
    # Reload model if needed (to ensure clean state)
    if model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model

# --------------------------------------------------
# 4. Tokenization
# --------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset for training."""
    print("\nTokenizing dataset...")
    
    def tokenize_function(example):
        tokenized = tokenizer(
            example["prompt"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )
        # Set labels same as input_ids for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize the dataset with batching for speed
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["question", "answer", "prompt"]
    )
    
    print(f"Tokenized {len(tokenized_datasets['train'])} training examples")
    print(f"Tokenized {len(tokenized_datasets['validation'])} validation examples")
    
    return tokenized_datasets

# --------------------------------------------------
# 5. Training Setup and Execution
# --------------------------------------------------

def train_model(model, tokenized_datasets, output_dir="./fashion_advisor_model"):
    """Set up training arguments and train the model."""
    print("\nSetting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,     # Small batch size for memory constraints
        gradient_accumulation_steps=8,     # Accumulate gradients to simulate larger batch
        optim="adamw_torch",
        fp16=True,                         # Use mixed precision
        gradient_checkpointing=True,       # Save memory
        save_total_limit=1,                # Keep only the last checkpoint
        logging_steps=10,
        save_strategy="epoch",             # Save at end of each epoch
        evaluation_strategy="no",          # Can change to "epoch" if desired
        learning_rate=2e-4,
        num_train_epochs=3
    )
    
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    return trainer, output_dir

# --------------------------------------------------
# 6. Save Model and Test
# --------------------------------------------------

def save_model(model, tokenizer, output_dir):
    """Save the fine-tuned model and tokenizer."""
    print(f"\nSaving model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully")

def test_model(model_path, test_prompt=None):
    """Test the fine-tuned model with a sample prompt."""
    print(f"\nTesting model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    # Create a text generation pipeline
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=512, 
        do_sample=True, 
        temperature=0.7
    )
    
    # Use default prompt if none provided
    if test_prompt is None:
        test_prompt = "Answer the question for an Indian college student's fashion doubt: What type of outfit is recommended for a rainy day?\nResponse:"
    
    print(f"Test prompt: {test_prompt}")
    
    # Generate a response
    response = generator(test_prompt)
    print(f"Generated response: {response[0]['generated_text']}")
    
    return response

# --------------------------------------------------
# 7. Main Function
# --------------------------------------------------

def main():
    """Main execution function."""
    print("=== Indian College Student Fashion Advisor Training ===")
    
    # 1. Load and prepare data
    file_path = "data-3 (2).docx"  # Update with your data file path
    df = load_data_from_docx(file_path)
    train_csv_path, val_csv_path = clean_and_prepare_data(df)
    
    # 2. Format and load dataset
    dataset = format_and_load_dataset(train_csv_path, val_csv_path)
    
    # 3. Setup model and tokenizer
    checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
    model, tokenizer = setup_model_and_tokenizer(checkpoint)
    
    # 4. Prepare model for training
    model = prepare_model_for_training(model, checkpoint)
    
    # 5. Tokenize dataset
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)
    
    # 6. Train model
    trainer, output_dir = train_model(model, tokenized_datasets, "./fashion_advisor_model")
    
    # 7. Save model
    save_model(model, tokenizer, output_dir)
    
    # 8. Test model
    test_model(output_dir)
    
    print("\n=== Training and testing completed successfully ===")

if __name__ == "__main__":
    main()
