#!/usr/bin/env python
# coding: utf-8

# ***
# Approach 1: End-to-End Code for Fine-Tuning DeepSeek R1 1.5B for Domain-Specific Text Generation
# ***


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Define the model name
#model_name = "deepseek-ai/deepseek-llm-7b-chat"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

#the following will exceed the RAM 32GB
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load pre-trained model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model)


# Move model to GPU if available
device =  "cpu"
model = model.to(device)

def generate_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states


    last_hidden_state = hidden_states[-1]


    embedding = last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()




device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

text = "CCCCO"
embedding = generate_embedding(text, model, tokenizer)
print("embedding shape:", embedding.shape)
print(embedding)


# One test before fine-tuning


max_length=80
def generate_text(prompt, max_length=max_length):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_k=50, top_p=0.9)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "provide one example SMILES that contains group \"COC\"."
output = generate_text(prompt)
print(output)


from torch.utils.data import Dataset, DataLoader

class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=max_length):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        inputs = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze()

df = pd.read_csv("../DATA/zinc100k_smiles.csv")
smiles_list = df['smiles'].tolist()


#smiles_list=["CCO", "CC(=O)O", "C1=CC=CC=C1", "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1", "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1", "CCCCC(=O)NC(=S)Nc1ccccc1C(=O)N1CCOCC1", "CCOC(=O)C(C)(C)c1nc(-c2ccccc2)no1"]
# 创建数据集和数据加载器
dataset = SmilesDataset(smiles_list, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


from datasets import Dataset


# Create a Hugging Face dataset
dataset = Dataset.from_dict({"SMILES": smiles_list})
print(len(dataset['SMILES']))



def preprocess_function(examples):
    inputs = tokenizer(
        examples["SMILES"], truncation=True, padding="max_length", max_length=max_length
    )

    # Labels must be a shifted version of input_ids for causal LM training
    inputs["labels"] = inputs["input_ids"].copy()
    #print(inputs)
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Apply LoRA for Efficient Fine-Tuning



from peft import get_peft_model, LoraConfig, TaskType

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  "lm_head"],
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj",],
    #target_modules=["q_proj", "k_proj",  ],
    lora_dropout=0.05,
    bias="none",
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)


# Configure Training Hyperparameters
"""
from trl import SFTTrainer
from transformers import TrainingArguments


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_dataset,
    dataset_text_field = "SMILES",
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 100,
        max_steps = 100,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16=True,  # Enable mixed precision training
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 2025,
        output_dir='outputs'
    ),
)

"""

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Adjusted for GPU memory limitations
    gradient_accumulation_steps=8,  # To simulate a larger batch size
    warmup_steps=100,
    max_steps=100,
    num_train_epochs = 3,
    learning_rate=2e-4,
    fp16=True,  # Enable mixed precision training
    logging_steps=10,
    output_dir="outputs",
    report_to="none",
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)



# Move model to CPU to free memory before training
model = model.to("cpu")


# Free up memory before training
import torch
import gc

gc.collect()  # Garbage collection
torch.cuda.empty_cache()  # Clears CUDA cache

# Optimize model with torch.compile (improves execution speed)
model = torch.compile(model)


# Move model back to GPU for training
model = model.to("cuda")
#model = model.to("cpu")
# Start training
trainer.train()


# Save the Fine-Tuned Model


model_path = "fine-tuned-deepseek-r1-1.5b-zinc100K"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to CPU (or GPU if needed)
# device = "cuda" if torch.cuda.is_available() else "cpu"
model.to("cpu")  # Keeping it on CPU for now
text = "CCCCO"
embedding = generate_embedding(text, model, tokenizer)
print("embedding shape:", embedding.shape)
print(embedding)


# Test the Fine-Tuned Model


def generate_text(prompt, max_length=max_length):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_k=50, top_p=0.9)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "provide one example SMILES that contains group \"COC\"."
output = generate_text(prompt)
print(output)

