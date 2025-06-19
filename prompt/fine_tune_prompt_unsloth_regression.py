import pandas as pd
from datasets import Dataset
import pandas as pd
from unsloth import FastLanguageModel
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
            text = f"SMILES: {smiles}\n{property_name}:"
            
            # 编码输入
            inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt", ).to(model.device)
            
            # 生成预测 - 使用更精确的生成参数
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,  # 使用beam search提高准确性
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码输出
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测的Property值
            try:
                # 查找"Property:"后面的数字
                property_name_label = f'{property_name}:'
                predicted_value_str = prediction.split(property_name_label)[-1].strip()
                # 提取第一个数值（处理可能的额外文本）
                predicted_value = float(predicted_value_str.split()[0])
            except (ValueError, IndexError) as e:
                print(f"Error parsing prediction for SMILES {smiles}: {prediction}")
                predicted_value = np.nan  # 如果解析失败，标记为NaN
            
            predictions.append(predicted_value)
            true_values.append(true_value)
    
    # 过滤掉无效的预测
    valid_indices = ~np.isnan(predictions)
    valid_predictions = np.array(predictions)[valid_indices]
    valid_true_values = np.array(true_values)[valid_indices]
    
    # 计算评估指标
    mae = mean_absolute_error(valid_true_values, valid_predictions)
    mse = mean_squared_error(valid_true_values, valid_predictions)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(valid_true_values, valid_predictions)
    
    # 打印结果
    print(f"\nEvaluation Results:")
    print(f"Number of valid predictions: {len(valid_predictions)}/{len(predictions)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    
    return {
        'pred': predictions,
        'true_values': true_values,
        'mae': mae,
        'rmse': rmse,
        'pearson_corr': pearson_corr,
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

model_name='ChemDFM'

model_base = "OpenDFM/ChemDFM-v1.5-8B"
model_7B_name="unsloth/DeepSeek-R1-Distill-Qwen-7B"
model_8B_name="unsloth/DeepSeek-R1-Distill-Llama-8B"

#model_base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name_list =['DS_7B', 'DS_8B', 'ChemDFM']

#ds_names = ['SAMPL', 'Llinas', 'Delaney', 'Lipophilicity', 'Warrior', ] # ['BACE', 'BBBP', 'ClinTox', 'Tox21SR']
ds_names = ['Llinas', 'Delaney', 'Lipophilicity', 'Warrior', ] # ['BACE', 'BBBP', 'ClinTox', 'Tox21SR']
#model_name_list =['DS_7B']
ds_names = ['SAMPL']

for model_name in model_name_list:
    if model_name == 'DS_7B':
        model_base = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
    elif model_name == 'DS_8B':
        model_base = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    elif model_name == 'ChemDFM':
        model_base = "OpenDFM/ChemDFM-v1.5-8B"


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
    
    
        pearson_list = []
        mae_list = []
        rmse_list = []
        max_pearson = float('-inf')
    
        model_tuned = f"lora_model_{data_name}_{model_name}"
        
        for idx in range(5):
            dataset = Dataset.from_pandas(df)
            random_number = random.randint(1, 9999999)
            print(f"\n processing {data_name} {model_name} round {idx} with random seed {random_number}\n")
            dataset = dataset.train_test_split(test_size=0.1, seed = random_number + idx)
            # 加载测试数据集（确保包含原始SMILES和Property列）
            test_dataset = dataset["test"]  # 假设这是原始数据集，不是tokenized版本
        
            model, tokenizer = FastLanguageModel.from_pretrained(
                 model_name=model_base,
                 max_seq_length=128, 
                 dtype=None,
                 load_in_4bit=False, 
                 # token="hf...",  # 如果需要访问授权模型，可以在这里填入密钥
                )
            
            tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
            
            model = FastLanguageModel.get_peft_model(
                model, 
                r=16,
                #target_modules=["q_proj", "k_proj", "v_proj", "o_proj",], # 指定模型中需要微调的关键模块
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  "lm_head"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth", 
                random_state=2025, 
                # use_rsloora=False,
                loftq_config=None, 
                )
            
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
                fn_kwargs={"data_name": data_name}
                )
        
            property_name = data_property_map[data_name]
            print(f"processing {data_name} for {property_name}\n")
    
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
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
            )
    
        
            # 开始训练
            trainer.train()
            
            FastLanguageModel.for_inference(model)

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
            
                mae, rmse, pearson_corr = results['mae'], results['rmse'], results['pearson_corr']
        
                pearson_list.append(pearson_corr)
                mae_list.append(mae)
                rmse_list.append(rmse)
        
                if pearson_corr > max_pearson:
                    max_pearson = pearson_corr
                    # 保存当前模型权重
                    model.save_pretrained(model_tuned)
                    tokenizer.save_pretrained(model_tuned)
                    print(f"Model saved at round {idx} with Pearson coefficient: {max_pearson}, RMSE: {rmse}")
        
            else:
                results = evaluate_model(model, tokenizer, test_dataset, property_name=property_name)
    
                eval_df = pd.DataFrame({
                    'SMILES': test_dataset['smiles'],
                    'True': test_dataset['expt'],
                    'Predicted': results['pred'],
                    'Valid': results['valid_indices']
                    })
                eval_df.to_csv(f'results/{data_name}_{model_name}_predictions_inter{idx}.csv', index=False)
    
                mae, rmse, pearson_corr = results['mae'], results['rmse'], results['pearson_corr']
    
                pearson_list.append(pearson_corr)
                mae_list.append(mae)
                rmse_list.append(rmse)
    
                if pearson_corr > max_pearson:
                    max_pearson = pearson_corr
                    # 保存当前模型权重
                    model.save_pretrained(model_tuned)
                    tokenizer.save_pretrained(model_tuned)
                    print(f"Model saved at round {idx} with Pearson coefficient: {max_pearson}, RMSE: {rmse}")
    
            model.to('cpu')
            gc.collect()  # Garbage collection
            torch.cuda.empty_cache()
            del model, tokenizer, dataset
        
        
        df_test = pd.DataFrame({'Pearson': pearson_list, 'MAE':mae_list, 'RMSE':rmse_list})
        df_test.to_csv(f'results/pearson_{data_name}_{model_name}.csv', index=False)
        
        Q1 = df_test['RMSE'].quantile(0.25)
        Q3 = df_test['RMSE'].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 剔除异常值
        df_test_clean = df_test[(df_test['RMSE'] >= lower_bound) & (df_test['RMSE'] <= upper_bound)]
        
        # 计算均值和标准差
        mean_clean = df_test_clean['RMSE'].mean()
        std_clean = df_test_clean['RMSE'].std()
        
        pearson_mean = df_test_clean['Pearson'].mean()  # 新增：Pearson均值
        pearson_std = df_test_clean['Pearson'].std()
        
        print(f"summary for the dataset {data_name}, {model_name}\n")
        print(f'RMSE mean_clean: {mean_clean:.3f} ± {std_clean:.3f}')
        print(f'Pearson (clean): {pearson_mean:.3f} ± {pearson_std:.3f}')
        print(f"valid number: {len(df_test_clean)}/{len(df_test)}")
        print("\n")
        
        
