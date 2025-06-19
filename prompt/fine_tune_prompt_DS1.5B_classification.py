import pandas as pd
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm  # 用于显示进度条
import torch
import gc
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import re
from transformers import LogitsProcessor

class BooleanConstraintProcessor(LogitsProcessor):
    def __init__(self, tokenizer, true_token_id, false_token_id):
        self.tokenizer = tokenizer
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        # 获取所有无效token的mask
        self.all_token_ids = set(range(tokenizer.vocab_size))
        self.valid_token_ids = {true_token_id, false_token_id}

    def __call__(self, input_ids, scores):
        # 强制将非True/False的token概率设为负无穷
        for token_id in self.all_token_ids - self.valid_token_ids:
            scores[:, token_id] = -float('inf')
        return scores



def preprocess_function(df, data_name=None):
    if data_name not in data_property_map:
        raise ValueError(f"Unknown data_name: {data_name}. Available options: {list(data_property_map.keys())}")
    
    property_name = data_property_map[data_name]

    # 将SMILES和Property组合成文本
    texts = [f"SMILES: {smiles}\n{property_name}: {expt}" for smiles, expt in zip(df['smiles'], df['expt'])]
    
    # 对文本进行tokenize
    model_inputs = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # 创建标签 - 这里我们使用输入ID作为标签，因为这是一个语言模型
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# 设置LoRA配置
lora_config = LoraConfig(
    r=8,  # LoRA的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    warmup_steps = 100,
    max_steps = 200,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # 如果GPU支持的话
)



def evaluate_model(model, tokenizer, test_dataset, property_name):
    model.eval()  # 设置模型为评估模式
    predictions = []
    true_values = []
    
    # 获取原始SMILES和Property值
    smiles_list = test_dataset['smiles']
    true_value_list = test_dataset['expt']
    
    with torch.no_grad():
        for smiles, true_value in tqdm(zip(smiles_list, true_value_list), total=len(smiles_list)):
            # 准备输入文本
            text = f"SMILES: {smiles}\n {property_name}:"
            
            # 编码输入
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # 生成预测 - 使用更精确的生成参数
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                num_beams=3,  # 使用beam search提高准确性
                logits_processor=[BooleanConstraintProcessor(tokenizer, true_token_id, false_token_id)],
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码输出
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                # 方法1：先尝试匹配true/false（不区分大小写）
                bool_match = re.search(r'\b(true|false)\b', prediction, flags=re.IGNORECASE)
                if bool_match:
                    predicted_value = 1 if bool_match.group(0).lower() == 'true' else 0
                else:
                    # 方法2：如果没有true/false，则提取第一个数值
                    num_match = re.search(r'[-+]?\d*\.\d+|\d+', prediction)
                    if num_match:
                        predicted_value = float(num_match.group(0))
                    else:
                        # 方法3：如果都没有，尝试从"Property:"后提取
                        property_name_label = f'{property_name}:'
                        predicted_value_str = prediction.split(property_name_label)[-1].strip().lower()
                        if predicted_value_str == 'true':
                            predicted_value = 1
                        elif predicted_value_str == 'false':
                            predicted_value = 0
                        else:
                            try:
                                predicted_value = float(predicted_value_str.split()[0])
                            except (ValueError, IndexError):
                                predicted_value = np.nan
                                print(f"Cannot parse prediction for SMILES {smiles}: {prediction}")
            
            except Exception as e:
                print(f"Error parsing prediction for SMILES {smiles}: {prediction} - {str(e)}")
                predicted_value = np.nan
            print(prediction,"\n", predicted_value)
            
            predictions.append(predicted_value)
            true_values.append(true_value)
    
    # 过滤掉无效的预测
    valid_indices = ~np.isnan(predictions)
    valid_predictions = np.array(predictions)[valid_indices]
    valid_true_values = np.array(true_values)[valid_indices]
    
    # 计算评估指标
    valid_predictions = np.clip(valid_predictions, 0, 1)
    auc = roc_auc_score(valid_true_values, valid_predictions)
    accuracy = accuracy_score(valid_true_values, valid_predictions)
    precision = precision_score(valid_true_values, valid_predictions, average='binary')  # 二分类
    recall = recall_score(valid_true_values, valid_predictions, average='binary')  # 二分类
    f1 = f1_score(valid_true_values, valid_predictions, average='binary')  # 二分类
    conf_matrix = confusion_matrix(valid_true_values, valid_predictions)

    print(f"\nEvaluation Results:")
    print(f"Number of valid predictions: {len(valid_predictions)}/{len(predictions)}")
    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    return {
        'pred': predictions,
        'true_values': true_values,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-scaore': f1,
        'valid_indices': valid_indices
    }
 


data_property_map = {
    "SAMPL": "hydration free energy",
    "Lipophilicity": "logD",
    "Llinas": 'logS',
    "Delaney": 'logS',
    "Warrior": 'pKa',
    "BACE": 'inhibitor activity',
    "BBBP": 'activity',
    "ClinTox": 'toxicity',
    "Tox21SR": 'toxicity',
}

model_name='DS1.5B'

model_base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

ds_names = ['SAMPL', 'Delaney', 'Lipophilicity', 'Warrior',] 
ds_names = ['BACE', 'BBBP', 'ClinTox', 'Tox21SR']
ds_names = ['Tox21SR']

for data_name in ds_names:
    if data_name == 'SAMPL':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/SAMPL.csv')
    elif data_name == 'Delaney':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/delaney-processed.csv')
        df = df.rename(columns={"measured log solubility in mols per litre": "expt"})
    elif data_name == 'Lipophilicity':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/Lipophilicity.csv')
        df = df.rename(columns={"exp": "expt"})
    elif data_name == 'Warrior':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/DataWarrior.csv')
        df = df.rename(columns={"pKa": "expt"})
    elif data_name == 'BACE':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/bace.csv')
        df = df.rename(columns={"SMILES": "smiles", "label": "expt"})
    elif data_name == 'BBBP':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/BBBP.csv')
        df = df.rename(columns={"p_np": "expt"})
    elif data_name == 'ClinTox':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/clintox.csv')
        df = df.rename(columns={"CT_TOX": "expt"})
    elif data_name == 'Tox21SR':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/tox21.csv')
        df = df.dropna(subset=["SR-ARE"])
        df = df.rename(columns={"SR-ARE": "expt"})
    elif data_name == 'Llinas':
        df = pd.read_csv('~/deepseek/KFLM2/dataset/aqsol_llinas_train.csv')
        df = df.rename(columns={"y": "expt"})
        test1_df = pd.read_csv('~/deepseek/KFLM2/dataset/llinas2020_set1_test.csv')
        test1_df = test1_df.rename(columns={"y": "expt"})

        test2_df = pd.read_csv('~/deepseek/KFLM2/dataset/llinas2020_set2_test.csv')
        test2_df = test2_df.rename(columns={"y": "expt"})

    df['expt'] = df['expt'].astype(bool)
    print(df.head())

    auc_list = []
    accuracy_list = []
    precision_list = []
    max_pearson = float('-inf')


    model_tuned = f"lora_model_{data_name}_{model_name}"
    
    for idx in range(10):
        dataset = Dataset.from_pandas(df)
        random_number = random.randint(1, 9999999)
        dataset = dataset.train_test_split(test_size=0.1, seed = random_number + idx)
        # 加载测试数据集（确保包含原始SMILES和Property列）
        test_dataset = dataset["test"]  # 假设这是原始数据集，不是tokenized版本
    
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(model_base)
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
        # 应用预处理
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            fn_kwargs={"data_name": data_name}
            )
    
        property_name = data_property_map[data_name]
        print(f"processing {data_name} for {property_name}\n")
        # 应用LoRA
        model = get_peft_model(model, lora_config)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
        )
    
        # 开始训练
        trainer.train()
        
        # 获取True/False对应的token ID
        true_token_id = tokenizer.encode("True", add_special_tokens=False)[0]
        false_token_id = tokenizer.encode("False", add_special_tokens=False)[0]
        # 运行评估
        if data_name == 'Llinas':
            test_df = pd.concat([test1_df, test2_df], axis=0)

            test_df = test_df.reset_index(drop=True)

            test_dataset_llinas = Dataset.from_pandas(test_df)
 
            results = evaluate_model(model, tokenizer, test_dataset_llinas, property_name=property_name)
        
            eval_df = pd.DataFrame({
                'SMILES': test_dataset_llinas['smiles'],
                'True': test_dataset_llinas['expt'],
                'Predicted': results['pred'],
                'Valid': results['valid_indices']
                })
            eval_df.to_csv(f'results/{data_name}_{model_name}_predictions_inter{idx}.csv', index=False)
        
            auc, accuracy, precision = results['auc'], results['accuracy'], results['precision'] 
            auc_list.append(auc)
            accuracy_list.append(accuracy)
            precision_list.append(precision)

    
            if pearson_corr > max_pearson:
                max_pearson = pearson_corr
                # 保存当前模型权重
                model.save_pretrained(model_tuned)
                tokenizer.save_pretrained(model_tuned)
                print(f"Model saved at round {idx} with Pearson coefficient: {max_pearson}")
    
        else:
            results = evaluate_model(model, tokenizer, test_dataset, property_name=property_name)

            eval_df = pd.DataFrame({
                'SMILES': test_dataset['smiles'],
                'True': test_dataset['expt'],
                'Predicted': results['pred'],
                'Valid': results['valid_indices']
                })
            eval_df.to_csv(f'results/{data_name}_{model_name}_predictions_inter{idx}.csv', index=False)

            auc, accuracy, precision = results['auc'], results['accuracy'], results['precision']

            auc_list.append(auc)
            accuracy_list.append(accuracy)
            precision_list.append(precision)

            if auc > max_pearson:
                max_pearson = auc
                # 保存当前模型权重
                model.save_pretrained(model_tuned)
                tokenizer.save_pretrained(model_tuned)
                print(f"Model saved at round {idx} with AUC coefficient: {max_pearson}")


    model.to('cpu')
    gc.collect()  # Garbage collection
    torch.cuda.empty_cache()
    del model, tokenizer, dataset
    
    
    df_test = pd.DataFrame({'AUC': auc_list, 'accuracy':accuracy_list, 'precision':precision_list})
    df_test.to_csv(f'results/auc_{data_name}_{model_name}.csv', index=False)

    Q1 = df_test['AUC'].quantile(0.25)
    Q3 = df_test['AUC'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_test_clean = df_test[(df_test['AUC'] >= lower_bound) & (df_test['AUC'] <= upper_bound)]

    mean_clean = df_test_clean['AUC'].mean()
    std_clean = df_test_clean['AUC'].std()

    accuracy_mean = df_test_clean['accuracy'].mean()  # ??:Accuracy??
    accuracy_std = df_test_clean['accuracy'].std()

    print(f"summary for the dataset {data_name}, {model_name}\n")
    print(f'AUC mean_clean: {mean_clean:.3f} ± {std_clean:.3f}')
    print(f'Accuracy (clean): {accuracy_mean:.3f} ± {accuracy_std:.3f}')
    print(f"valid number: {len(df_test_clean)}/{len(df_test)}")
    print("\n")

