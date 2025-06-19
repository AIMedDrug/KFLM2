#!/usr/bin/env python
# coding: utf-8

# ***
# Approach 1: End-to-End Code for Fine-Tuning DeepSeek R1 1.5B for Domain-Specific Text Generation
# ***


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
import pandas as pd
from unsloth import FastLanguageModel  
# Define the model name
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

#the following will exceed the RAM 32GB
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Load pre-trained model & tokenizer
#model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B"
model_name = "OpenDFM/ChemDFM-v1.5-8B"

#model_name="unsloth/DeepSeek-R1-Distill-Llama-8B"
max_seq_length=128
model, tokenizer = FastLanguageModel.from_pretrained(       
     model_name=model_name,  # 指定要加载的模型名称       
     max_seq_length=max_seq_length,  # 使用前面设置的最大长度       
     dtype=None,  # 使用前面设置的数据类型       
     load_in_4bit=False,  # 使用4位量化       
     # token="hf...",  # 如果需要访问授权模型，可以在这里填入密钥  
    )

print(model)


def generate_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states


    last_hidden_state = hidden_states[-1]


    embedding = last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()


FastLanguageModel.for_inference(model)

text = "CCCCO"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=120,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0])


text = "CCCCO"
embedding = generate_embedding(text, model, tokenizer)
print("embedding shape:", embedding.shape)
print(embedding)



# One test before fine-tuning
device = "cuda" if torch.cuda.is_available() else "cpu"


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


# Step 3: Generate a Hypothetical Domain-Specific Document


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

df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv")
df['smiles'] = df['smiles'].apply(lambda s: s.replace('\n', ''))
smiles_list = df['smiles'].tolist()

#smiles_list=["CCO", "CC(=O)O", "C1=CC=CC=C1", "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1", "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1", "CCCCC(=O)NC(=S)Nc1ccccc1C(=O)N1CCOCC1", "CCOC(=O)C(C)(C)c1nc(-c2ccccc2)no1"]
# 创建数据集和数据加载器
dataset = SmilesDataset(smiles_list, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Step 4: Convert the Text Data into a Hugging Face Dataset

from datasets import Dataset

# Create a Hugging Face dataset
dataset = Dataset.from_dict({"SMILES": smiles_list})
print(len(dataset['SMILES']))


# Step 5: Tokenization with Labeling
def preprocess_function(examples):
    examples["SMILES"] = [smiles + " <EOS>" for smiles in examples["SMILES"]]
    inputs = tokenizer(
        examples["SMILES"], truncation=True, padding="max_length", max_length=max_length
    )

    # Labels must be a shifted version of input_ids for causal LM training
    inputs["labels"] = inputs["input_ids"].copy()
    #print(inputs)
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Step 6: Apply LoRA for Efficient Fine-Tuning


FastLanguageModel.for_training(model)      

model = FastLanguageModel.get_peft_model(   
    model,  # 传入已经加载好的预训练模型       
    r=16,  # 设置 LoRA 的秩，决定添加的可训练参数数量       
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj",], # 指定模型中需要微调的关键模块          
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  "lm_head"],
    lora_alpha=16,  # 设置 LoRA 的超参数，影响可训练参数的训练方式       
    lora_dropout=0,  # 设置防止过拟合的参数，这里设置为 0 表示不丢弃任何参数       
    bias="none",  # 设置是否添加偏置项，这里设置为 "none" 表示不添加       
    use_gradient_checkpointing="unsloth",  # 使用优化技术节省显存并支持更大的批量大小       
    random_state=3407,  # 设置随机种子，确保每次运行代码时模型的初始化方式相同       
    # use_rsloora=False,  # 设置是否使用 Rank Stabilized LoRA 技术，这里设置为 False 表示不使用       
    loftq_config=None,  # 设置是否使用 LoftQ 技术，这里设置为 None 表示不使用   
    )      

from trl import SFTTrainer  # 导入 SFTTrainer，用于监督式微调   
from transformers import TrainingArguments  # 导入 TrainingArguments，用于设置训练参数   
from unsloth import is_bfloat16_supported  # 导入函数，检查是否支持 bfloat16 数据格式     

trainer = SFTTrainer(    
    model=model,  # 传入要微调的模型       
    tokenizer=tokenizer,  # 传入 tokenizer，用于处理文本数据       
    train_dataset=dataset,  # 传入训练数据集       
    dataset_text_field="SMILES",  # 指定数据集中文本字段的名称       
    max_seq_length=max_seq_length,  # 设置最大序列长度       
    dataset_num_proc=2,  # 设置数据处理的并行进程数       
    packing=False,  # 是否启用打包功能（这里设置为 False，打包可以让训练更快，但可能影响效果）       
    args=TrainingArguments(           
          per_device_train_batch_size=2,  # 每个设备（如 GPU）上的批量大小           
          gradient_accumulation_steps=4,  # 梯度累积步数，用于模拟大批次训练           
          warmup_steps=100,  # 预热步数，训练开始时学习率逐渐增加的步数           
          max_steps=100,  # 最大训练步数           
          learning_rate=2e-4,  # 学习率，模型学习新知识的速度           
          fp16=not is_bfloat16_supported(),  # 是否使用 fp16 格式加速训练（如果环境不支持 bfloat16）           
          bf16=is_bfloat16_supported(),  # 是否使用 bfloat16 格式加速训练（如果环境支持）           
          logging_steps=1,  # 每隔多少步记录一次训练日志           
          optim="adamw_8bit",  # 使用的优化器，用于调整模型参数           
          weight_decay=0.01,  # 权重衰减，防止模型过拟合           
          lr_scheduler_type="linear",  # 学习率调度器类型，控制学习率的变化方式           
          seed=3407,  # 随机种子，确保训练结果可复现           
          output_dir="outputs",  # 训练结果保存的目录           
          report_to="none",  # 是否将训练结果报告到外部工具（如 WandB），这里设置为不报告   
              ),  
    )



# Free up memory before training
import torch
import gc

gc.collect()  # Garbage collection
torch.cuda.empty_cache()  # Clears CUDA cache


# Start training
trainer.train()


# Step 10: Save the Fine-Tuned Model


# Define the path where the fine-tuned model is saved
model_save_path = "fine-tuned-chemdfm-8b"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


# Step 11: Load the Fine-Tuned Model

FastLanguageModel.for_inference(model)  

# Move model to CPU (or GPU if needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
text = "CCCCO"
embedding = generate_embedding(text, model, tokenizer)
print("embedding shape:", embedding.shape)
print(embedding)


# Step 12: Test the Fine-Tuned Model



def generate_text(prompt, max_length=max_length):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_k=50, top_p=0.9)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "provide one example SMILES that contains group \"COC\"."
output = generate_text(prompt)
print(output)

