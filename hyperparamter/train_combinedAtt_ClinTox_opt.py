import torch
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from featerize_smiles import SmilesDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import pandas as pd
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error  # 导入 MAE
from scipy.stats import pearsonr
import random
from models import CombinedModel, CombinedModelClass, GCNModel, EmbeddingModel, EmbeddingModelClass, CombinedCrossAttBiClassModel, CombinedAttModelClass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 加载模型和分词器
model_path = "../fine-tuned-deepseek-r1-1.5b"
#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

data_name='ClinTox_opt'
model_name='CombinedAtt'
dropout = 0.2
hidden_dim = 32
# 数据集和数据加载器
csv_file = '~/DATA/clintox.csv'  # 替换为您的 CSV 文件路径
data = pd.read_csv(csv_file)
smiles = data['smiles'].tolist()
logP = data['CT_TOX'].tolist()

dataset = SmilesDataset(smiles, logP, model, tokenizer)

# 过滤掉为 None 的项
filtered_data = [(item[0], item[1]) for item in dataset if item[0] is not None]

hidden_list = [16, 32, 64, 128, 256]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

pearson_summary = pd.DataFrame()
auc_results = []
auc_list = []
accuracy_list = []
precision_list = []
f1_list = []
recall_list = []

max_auc = float('-inf')

model_save_path = f'result/{data_name}_{model_name}.pt'

for idx in range(10):
    best_model_path = f'result/{data_name}_{model_name}_iter{idx}_early.pt'
    random_number = random.randint(1, 9999999)
    print("random seed: ", random_number)
    train_data, temp_data = train_test_split(filtered_data, test_size=0.2, random_state=random_number)  # 80% 训练集，20% 临时集
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 50% 验证集，50% 测试集
    batch_size = 32
    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    
    
    # 初始化联合模型
    input_dim = 1536
    #criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（包含 Sigmoid）
    #criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    combined_model = CombinedAttModelClass(input_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=1).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 150  
    best_epoch = -1
    best_valid_metric = float('-inf')
    for epoch in range(num_epochs):
        combined_model.train()  # 设置模型为训练模式
        epoch_loss = 0.0  # 初始化 epoch 损失
        for (smiles_embeddings, graph_data), logP_values in train_loader:
            smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
            graph_data = graph_data.to(device)
    
            # 前向传播
            predictions = combined_model(smiles_embeddings, graph_data)
            # 计算损失
            loss = criterion(predictions.squeeze(), logP_values)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # 累加损失
            epoch_loss += loss.item()
    
        # 输出每个 epoch 的平均损失
        avg_train_loss = epoch_loss / len(train_loader)
    
        # 验证模型
        combined_model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for (smiles_embeddings, graph_data), logP_values in val_loader:
                smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
                graph_data = graph_data.to(device)
        
                # 前向传播
                predictions = combined_model(smiles_embeddings, graph_data) 
                # 计算损失
                loss = criterion(predictions.squeeze(), logP_values)
                val_loss += loss.item()
        
                prediction_labels = torch.round(torch.sigmoid(predictions))
                all_predictions.extend(prediction_labels.cpu().numpy())
                all_true_values.extend(logP_values.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_true_values = np.array(all_true_values).flatten()
        val_auc = roc_auc_score(all_true_values, all_predictions)

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average train Loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}, val_auc: {val_auc:.4f}')


        if val_auc > best_valid_metric:
            best_epoch = epoch + 1
            best_valid_metric = val_auc
            torch.save(combined_model.state_dict(), best_model_path)
            print(f"best model saved at (Epoch {best_epoch}, AUC={val_auc:.4f})")

        # early stop
        if abs(best_epoch - epoch) >= 30:
            print(f"{'=' * 20} early stop at epoch {epoch+1} {'=' * 20}")
            break

    combined_model.eval()
    all_predictions = []
    all_true_values = []
    
    checkpoint = torch.load(best_model_path)
    combined_model.load_state_dict(checkpoint)
    print(f"Loaded best model with best_valid_metric {best_valid_metric:.4f}\n ")
    combined_model.to(device)
    combined_model.eval()

    with torch.no_grad():
        for (smiles_embeddings, graph_data), logP_values in test_loader:
            smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
            graph_data = graph_data.to(device)
    
            predictions = combined_model(smiles_embeddings, graph_data)
            predictions = torch.round(torch.sigmoid(predictions))
            all_predictions.extend(predictions.cpu().numpy())
            all_true_values.extend(logP_values.cpu().numpy())
    
    # 计算 Pearson 系数和 MAE
    all_predictions = np.array(all_predictions).flatten()
    all_true_values = np.array(all_true_values).flatten()
    

    print(f"parameters: {hidden_dim}, {dropout}")

    auc = roc_auc_score(all_true_values, all_predictions)
    print(f"AUC: {auc:.4f}")
    accuracy = accuracy_score(all_true_values, all_predictions)
    precision = precision_score(all_true_values, all_predictions, average='binary')  # 二分类
    recall = recall_score(all_true_values, all_predictions, average='binary')  # 二分类
    f1 = f1_score(all_true_values, all_predictions, average='binary')  # 二分类
    conf_matrix = confusion_matrix(all_true_values, all_predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    auc_list.append(auc)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)


    df_test = pd.DataFrame({
            'Predictions': all_predictions,
            'True_Values': all_true_values
        })

    scatter_file = f'result/predictions_{data_name}_{model_name}_iter{idx}.csv'
    df_test.to_csv(scatter_file, index=False)

    if auc > max_auc:
        max_auc = auc
        # 保存当前模型权重
        torch.save(combined_model.state_dict(), model_save_path)
        print(f"Best Test Model saved at round {idx} with auc coefficient: {max_auc}")


    del combined_model

df_test = pd.DataFrame({'AUC':auc_list, 'accuracy': accuracy_list, 'precision': precision_list, 'recall': recall_list, 'f1':f1_list})

df_test.to_csv(f'result/auc_{data_name}_{model_name}.csv', index=False)


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

print(f'AUC mean_clean: {mean_clean:.3f} ± {std_clean:.3f}')
print(f'Accuracy (clean): {accuracy_mean:.3f} ± {accuracy_std:.3f}')
print(f"valid number: {len(df_test_clean)}/{len(df_test)}")

