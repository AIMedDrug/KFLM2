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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import random
from models import CombinedModel, CombinedSparseModel, CombinedAttModel, EarlyStopping
# 加载模型和分词器
model_path = "../fine-tuned-deepseek-r1-1.5b"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

data_name='Llinas_opt'
model_name='CombinedAtt'
dropout = 0.3
hidden_dim = 32

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
mae_list1 = []
rmse_list1 = []
mae_list2 = []
rmse_list2 = []

pearson_list = []
mae_list = []
rmse_list = []
max_pearson = float('-inf')

model_save_path = f'result/{data_name}_{model_name}.pt'

for idx in range(10):
    best_model_path = f'result/{data_name}_{model_name}_iter{idx}_early.pt'

    random_number = random.randint(1, 9999999)
    print("random seed: ", random_number)
    train_data, val_data = train_test_split(filtered_data, test_size=0.2, random_state=random_number)
    batch_size = 32
    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    
    
    # 初始化联合模型
    embedding_dim = 1536  # 根据模型的输出维度调整
    combined_model = CombinedAttModel(embedding_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 150
    best_epoch = -1
    best_valid_metric = float('inf')
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

        if avg_val_loss  < best_valid_metric:
            best_epoch = epoch + 1
            best_valid_metric = avg_val_loss
            torch.save(combined_model.state_dict(), best_model_path)
            print(f"best model saved at (Epoch {best_epoch}, loss={avg_val_loss:.4f})")
        # early stop
        if abs(best_epoch - epoch) >= 20:
            print(f"{'=' * 20} early stop at epoch {epoch+1} {'=' * 20}")
            break

    
    combined_model.eval()
    all_predictions = []
    all_true_values = []
    checkpoint = torch.load(best_model_path)
    combined_model.load_state_dict(checkpoint)
    print(f"Loaded best model with best_valid_metric: {best_valid_metric:.4f}\n ")
    combined_model.to(device)
    combined_model.eval()

    with torch.no_grad():
        for (smiles_embeddings, graph_data), logP_values in test_loader1:
            smiles_embeddings, logP_values = smiles_embeddings.to(device), logP_values.to(device)
            graph_data = graph_data.to(device)
    
            predictions = combined_model(smiles_embeddings, graph_data)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_values.extend(logP_values.cpu().numpy())
    
    # 计算 Pearson 系数和 MAE
    all_predictions = np.array(all_predictions).flatten()
    all_true_values = np.array(all_true_values).flatten()
    
    pearson_corr, _ = pearsonr(all_true_values, all_predictions)
    mae = mean_absolute_error(all_true_values, all_predictions)
    mse = mean_squared_error(all_true_values, all_predictions)
    rmse = np.sqrt(mse)
 
    print(f'Pearson Correlation Coefficient: {pearson_corr:.4f}')
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    pearson_list1.append(pearson_corr) 
    mae_list1.append(mae)
    rmse_list1.append(rmse)


    df_test = pd.DataFrame({
            'Predictions': all_predictions,
            'True_Values': all_true_values
        })
        
    scatter_file = f'result/predictions_{data_name}_{model_name}_iter{idx}_set1.csv'
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
    mae2 = mean_absolute_error(all_true_values, all_predictions)
    mse2 = mean_squared_error(all_true_values, all_predictions)
    rmse2 = np.sqrt(mse2)
    print(f'Pearson Correlation Coefficient of set2: {pearson_corr_1:.4f} at hidden: {hidden_dim}, dropout: {dropout}')
    print(f'Mean Absolute Error (MAE) and RMSE of set2: {mae2:.4f}, {rmse2:.4f}')
    pearson_corr_sum = pearson_corr + pearson_corr_1
    print(f'Pearson Correlation Coefficient of sum: {pearson_corr_sum:.4f} at hidden: {hidden_dim}, dropout: {dropout}')

    pearson_list2.append(pearson_corr_1)
    mae_list2.append(mae2)
    rmse_list2.append(rmse2)
    df_test = pd.DataFrame({
                'Predictions': all_predictions,
                'True_Values': all_true_values
            })
            
    # 将 DataFrame 保存为 CSV 文件
    scatter_file = f'result/predictions_{data_name}_{model_name}_iter{idx}_set2.csv'
    # 将 DataFrame 保存为 CSV 文件
    df_test.to_csv(scatter_file, index=False)

    rmse_sum = rmse + rmse2
    if pearson_corr_sum > max_pearson:
        max_pearson = pearson_corr_sum
        # 保存当前模型权重
        torch.save(combined_model.state_dict(), model_save_path)
        print(f"Model saved at round {idx} with Pearson coefficient: {max_pearson}, RMSE: {rmse_sum}")

    del combined_model


df_test = pd.DataFrame({'Pearson1': pearson_list1, 'MAE1':mae_list1, 'RMSE1':rmse_list1, 'Pearson2': pearson_list2, 'MAE2':mae_list2, 'RMSE2':rmse_list2})
df_test.to_csv(f'result/pearson_{data_name}_{model_name}.csv', index=False)

Q1 = df_test['RMSE1'].quantile(0.25)
Q3 = df_test['RMSE1'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 剔除异常值
df_test_clean = df_test[(df_test['RMSE1'] >= lower_bound) & (df_test['RMSE1'] <= upper_bound)]

# 计算均值和标准差
mean_clean = df_test_clean['RMSE1'].mean()
std_clean = df_test_clean['RMSE1'].std()

pearson_mean = df_test_clean['Pearson1'].mean()  # 新增：Pearson均值
pearson_std = df_test_clean['Pearson1'].std()

print(f'RMSE mean_clean: {mean_clean:.3f} ± {std_clean:.3f}')
print(f'Pearson (clean): {pearson_mean:.3f} ± {pearson_std:.3f}')
print(f"valid number: {len(df_test_clean)}/{len(df_test)}")

Q1 = df_test['RMSE2'].quantile(0.25)
Q3 = df_test['RMSE2'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 剔除异常值
df_test_clean = df_test[(df_test['RMSE2'] >= lower_bound) & (df_test['RMSE2'] <= upper_bound)]

# 计算均值和标准差
mean_clean = df_test_clean['RMSE2'].mean()
std_clean = df_test_clean['RMSE2'].std()

pearson_mean = df_test_clean['Pearson2'].mean()  # 新增：Pearson均值
pearson_std = df_test_clean['Pearson2'].std()

print(f'RMSE mean_clean: {mean_clean:.3f} ± {std_clean:.3f}')
print(f'Pearson (clean): {pearson_mean:.3f} ± {pearson_std:.3f}')
print(f"valid number: {len(df_test_clean)}/{len(df_test)}")
