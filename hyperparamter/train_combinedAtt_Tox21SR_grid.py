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
from models import CombinedModel, GCNModel, EmbeddingModel, EmbeddingModelClass, CombinedCrossAttBiClassModel, CombinedAttModelClass 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 加载模型和分词器
model_path = "../fine-tuned-deepseek-r1-1.5b"
#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

data_name='Tox21SR_multi_1'
model_name='CombinedAtt'
dropout = 0.2
hidden_dim = 16
# 数据集和数据加载器
csv_file = '~/DATA/tox21.csv'  # 替换为您的 CSV 文件路径
data = pd.read_csv(csv_file)
df = data.dropna(subset=["SR-ARE"])
smiles = df['smiles'].tolist()
labels = df['SR-ARE'].tolist()


dataset = SmilesDataset(smiles, labels, model, tokenizer)

# 过滤掉为 None 的项
filtered_data = [(item[0], item[1]) for item in dataset if item[0] is not None]

auc_list = []
accuracy_list = []
precision_list = []
f1_list = []
recall_list = []
auc_results = []

max_pearson = float('-inf')
model_save_path = f'result/{data_name}_{model_name}_best.pt'

hidden_list = [16, 32, 64, 128, 256]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for dropout in dropout_list:
    for hidden_dim in hidden_list:
        for idx in range(5):
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
            num_epochs = 100
            for epoch in range(num_epochs):
                combined_model.train()  # 设置模型为训练模式
                epoch_loss = 0.0  # 初始化 epoch 损失
                for (smiles_embeddings, graph_data), tox21_labels in train_loader:
                    smiles_embeddings, labels = smiles_embeddings.to(device), tox21_labels.to(device)
                    graph_data = graph_data.to(device)
            
                    # 前向传播
                    predictions = combined_model(smiles_embeddings, graph_data)
                    # 计算损失
                    loss = criterion(predictions.squeeze(), labels)
                    
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
                with torch.no_grad():
                    for (smiles_embeddings, graph_data), labels in val_loader:
                        smiles_embeddings, labels = smiles_embeddings.to(device), labels.to(device)
                        graph_data = graph_data.to(device)
                
                        # 前向传播
                        predictions = combined_model(smiles_embeddings, graph_data)
                        # 计算损失
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Average train Loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}')
            
            combined_model.eval()
            all_predictions = []
            all_true_values = []
            
            with torch.no_grad():
                for (smiles_embeddings, graph_data), labels in test_loader:
                    smiles_embeddings, labels = smiles_embeddings.to(device), labels.to(device)
                    graph_data = graph_data.to(device)
            
                    predictions = combined_model(smiles_embeddings, graph_data)
                    predictions = torch.sigmoid(predictions.squeeze())
                    all_predictions.append(predictions.cpu().numpy())
                    all_true_values.append(labels.cpu().numpy())
            
            # 计算 Pearson 系数和 MAE
            #all_predictions = np.array(all_predictions).flatten()
            #all_true_values = np.array(all_true_values).flatten()
            #print(all_predictions, all_true_values)
            all_predictions = np.concatenate(all_predictions, axis=0)  # 形状 [N, 12]
            all_true_values = np.concatenate(all_true_values, axis=0) 
        
            
            #print(all_predictions, all_true_values)
        
            auc = roc_auc_score(all_true_values, all_predictions)
            print(f"AUC: {auc:.4f}")
            all_predictions = (all_predictions >= 0.5).astype(np.int32)
            print(all_predictions, all_true_values)

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

        
            del combined_model
        
        
            auc_results.append((random_number, hidden_dim, dropout, auc))


max_corr = -np.inf
best_hidden_dim = None
best_dropout = None
best_num_layers = None
for random_number, hidden_dim, dropout, corr in auc_results:
    if corr > max_corr:
        max_corr = corr
        best_hidden_dim = hidden_dim
        best_dropout = dropout

# 打印结果
print(f"Best AUC: {max_corr}, at hidden_dim: {best_hidden_dim}, dropout: {best_dropout} ")

df = pd.DataFrame(auc_results, columns=['random_number','Hidden Dim', 'Dropout', 'AUC'])

# 保存为CSV
df.to_csv(f'auc_results_{data_name}_{model_name}_grid.csv', index=False)

        

df_test = pd.DataFrame({'auc':auc_list, 'accuracy': accuracy_list, 'precision': precision_list, 'recall': recall_list, 'f1':f1_list})
df_test.to_csv(f'result/auc_{data_name}_{model_name}_grid_summary_1.csv', index=False)

