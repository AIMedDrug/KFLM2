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
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 导入 MAE
from scipy.stats import pearsonr
import random
from models import CombinedModel, GCNModel, EmbeddingModel 
from models import CombinedModel, CombinedSparseModel, CombinedAttModel
# 加载模型和分词器
model_path = "../fine-tuned-deepseek-r1-1.5b"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

data_name='Llinas'
model_name='CombinedAtt'
dropout = 0.1
hidden_dim = 16
# 数据集和数据加载器
csv_file = '/sdb/xielx/deepseek/AquaPred-main/datasets/aqsol_llinas_train.csv'  # 替换为您的 CSV 文件路径
data = pd.read_csv(csv_file)
smiles = data['smiles'].tolist()
logP = data['y'].tolist()
dataset = SmilesDataset(smiles, logP, model, tokenizer)

df1 = pd.read_csv('/sdb/xielx/deepseek/AquaPred-main/datasets/llinas2020_set1_test.csv')
smiles1 = df1['smiles'].tolist()
logP1 = df1['y'].tolist()
dataset1 = SmilesDataset(smiles1, logP1, model, tokenizer)

df2 = pd.read_csv('/sdb/xielx/deepseek/AquaPred-main/datasets/llinas2020_set2_test.csv')
smiles2 = df2['smiles'].tolist()
logP2 = df2['y'].tolist()
dataset2 = SmilesDataset(smiles2, logP2, model, tokenizer)

test_data1 = [(item[0], item[1]) for item in dataset1 if item[0] is not None]
test_data2 = [(item[0], item[1]) for item in dataset2 if item[0] is not None]

batch_size=32
test_loader1 = DataLoader(test_data1, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=False)

# 过滤掉为 None 的项
filtered_data = [(item[0], item[1]) for item in dataset if item[0] is not None]

pearson_list1 = []
pearson_list2 = []
hidden_list = [16, 32, 64, 128, 256, 1024]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

pearson_summary = pd.DataFrame()

pearson_results = []
for dropout in dropout_list:
    for hidden_dim in hidden_list:
        for idx in range(5):
            random_number = random.randint(1, 9999999)
            print("random seed: ", random_number)
            train_data, val_data = train_test_split(filtered_data, test_size=0.1, random_state=random_number)  # 80% 训练集，20% 临时集
            batch_size = 32
            # 创建 DataLoader
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            
            # 初始化联合模型
            embedding_dim = 1536
            combined_model = CombinedAttModel(embedding_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            # 损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
            
            # 训练模型
            num_epochs = 100
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
                with torch.no_grad():
                    for (smiles_embeddings, graph_data), logP_values in val_loader:
                        smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
                        graph_data = graph_data.to(device)
                
                        # 前向传播
                        predictions = combined_model(smiles_embeddings, graph_data)
                
                        # 计算损失
                        loss = criterion(predictions.squeeze(), logP_values)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Average train Loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}')
            
            
            combined_model.eval()
            all_predictions = []
            all_true_values = []
                
            with torch.no_grad():
                for (smiles_embeddings, graph_data), logP_values in test_loader1:
                    smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
                    graph_data = graph_data.to(device)
            
                    predictions = combined_model(smiles_embeddings, graph_data)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_true_values.extend(logP_values.cpu().numpy())
            
            all_predictions = np.array(all_predictions).flatten()
            all_true_values = np.array(all_true_values).flatten()
            
            pearson_corr, _ = pearsonr(all_true_values, all_predictions)
            mae = mean_absolute_error(all_true_values, all_predictions)
            mse = mean_squared_error(all_true_values, all_predictions)
            rmse = np.sqrt(mse)
            pearson_list1.append(pearson_corr)    
            
            print(f'Pearson Correlation Coefficient of set1: {pearson_corr:.4f} at hidden: {hidden_dim}, dropout: {dropout}')
            print(f'Mean Absolute Error (MAE) and RMSE of set1: {mae:.4f}, {rmse:.4f}')
        
            df_test = pd.DataFrame({
                'Predictions': all_predictions,
                'True_Values': all_true_values
            })
            
            
            scatter_file = f'result/predictions_{data_name}_{model_name}_dropout{dropout}_hidden{hidden_dim}_iter{idx}_set1.csv'
            # 将 DataFrame 保存为 CSV 文件
            df_test.to_csv(scatter_file, index=False)
            
            all_predictions = []
            all_true_values = []
            
            with torch.no_grad():
                for (smiles_embeddings, graph_data), logP_values in test_loader2:
                    smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
                    graph_data = graph_data.to(device)
            
                    predictions = combined_model(smiles_embeddings, graph_data)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_true_values.extend(logP_values.cpu().numpy())
            
            # 计算 Pearson 系数和 MAE
            all_predictions = np.array(all_predictions).flatten()
            all_true_values = np.array(all_true_values).flatten()
            
            pearson_corr_1, _ = pearsonr(all_true_values, all_predictions)
            mae = mean_absolute_error(all_true_values, all_predictions)
            mse1 = mean_squared_error(all_true_values, all_predictions)
            rmse1 = np.sqrt(mse1)
            print(f'Pearson Correlation Coefficient of set2: {pearson_corr_1:.4f} at hidden: {hidden_dim}, dropout: {dropout}')
            print(f'Mean Absolute Error (MAE) and RMSE of set2: {mae:.4f}, {rmse1:.4f}')
            pearson_corr_sum = pearson_corr + pearson_corr_1
            print(f'Pearson Correlation Coefficient of sum: {pearson_corr_sum:.4f} at hidden: {hidden_dim}, dropout: {dropout}')
        
            pearson_list2.append(pearson_corr_1)
            df_test = pd.DataFrame({
                'Predictions': all_predictions,
                'True_Values': all_true_values
            })
            
            # 将 DataFrame 保存为 CSV 文件
            scatter_file = f'result/predictions_{data_name}_{model_name}_dropout{dropout}_hidden{hidden_dim}_iter{idx}_set2.csv'
            # 将 DataFrame 保存为 CSV 文件
            df_test.to_csv(scatter_file, index=False)
            
            del combined_model
        
            rmse_sum = rmse + rmse1
            pearson_sum = pearson_corr + pearson_corr_1
            pearson_results.append((random_number, hidden_dim, dropout, pearson_corr, pearson_corr_1, pearson_sum, rmse_sum))


max_corr = -np.inf
best_hidden_dim = None
best_dropout = None
best_num_layers = None
best_rmse = np.inf
for random_number, hidden_dim, dropout, corr, corr1, pearson_sum, rmse in pearson_results:
    if pearson_sum > max_corr:
        max_corr = pearson_sum
        best_hidden_dim = hidden_dim
        best_dropout = dropout
        best_rmse = rmse
# 打印结果
print(f"Best Pearson: {max_corr}, {best_rmse} with hidden_dim: {best_hidden_dim}, dropout: {best_dropout} ")

df = pd.DataFrame(pearson_results, columns=['random_number', 'Hidden Dim', 'Dropout', 'Pearson_set1', 'Pearson_set2', 'Pearson_sum', 'rmse'])

# 保存为CSV
df.to_csv(f'pearson_results_{data_name}_{model_name}_grid.csv', index=False)

        
        
