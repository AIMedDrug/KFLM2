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

data_name='Tox21_opt'
model_name='CombinedAtt'
dropout = 0.1
hidden_dim = 64
# 数据集和数据加载器
csv_file = '~/DATA/tox21.csv'  # 替换为您的 CSV 文件路径
df = pd.read_csv(csv_file)
def calculate_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')

# 读取数据集

# 处理缺失值，将空值替换为 NaN
labels = df[['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
             'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 
             'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']].fillna(0).values 
print(labels)
#df = data.dropna(subset=["SR-ARE"])
smiles = df['smiles'].tolist()


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
max_auc = float('-inf')

model_save_path = f'result/{data_name}_{model_name}.pt'


hidden_list = [16, 32, 64, 128, 256]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

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

    combined_model = CombinedAttModelClass(input_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=12).to(device)
    
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
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for (smiles_embeddings, graph_data), labels in val_loader:
                smiles_embeddings, labels = smiles_embeddings.to(device), labels.to(device)
                graph_data = graph_data.to(device)
        
                # 前向传播
                predictions = combined_model(smiles_embeddings, graph_data)
                # 计算损失
                loss = criterion(predictions.squeeze(), labels)
                val_loss += loss.item()
        
                predictions = torch.sigmoid(predictions.squeeze())
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average train Loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}')
    
        all_predictions = np.concatenate(all_predictions, axis=0)  # 形状 [N, 12]
        all_true_values = np.concatenate(all_true_values, axis=0)
        auc_scores_val = []
        for i in range(all_true_values.shape[1]):  # 遍历每个标签
            try:
                auc = roc_auc_score(all_true_values[:, i], all_predictions[:, i])
                auc_scores_val.append(auc)
            except ValueError:  # 处理全0或全1的标签
                auc_scores_val.append(np.nan)

        val_auc = np.nanmean(auc_scores_val)


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

    micro_auc = roc_auc_score(all_true_values.ravel(), all_predictions.ravel())
    print("Micro-AUC:", micro_auc)
    
    auc_scores = []
    for i in range(all_true_values.shape[1]):  # 遍历每个标签
        try:
            auc = roc_auc_score(all_true_values[:, i], all_predictions[:, i])
            auc_scores.append(auc)
        except ValueError:  # 处理全0或全1的标签
            auc_scores.append(np.nan)

    macro_auc = np.nanmean(auc_scores)  # 宏平均
    print("label AUC:", auc_scores)
    print("Macro-AUC:", macro_auc)

    binary_predictions = (all_predictions >= 0.5).astype(np.int32)
    #auc = roc_auc_score(all_true_values, binary_predictions,  average='macro')
    #print(f"AUC: {auc:.4f}")
    #print(binary_predictions, all_true_values)
    accuracy = accuracy_score(all_true_values, binary_predictions)
    precision = precision_score(all_true_values, binary_predictions, average='macro')  # 二分类
    recall = recall_score(all_true_values, binary_predictions, average='macro')  # 二分类
    f1 = f1_score(all_true_values, binary_predictions, average='macro')  # 二分类
    #conf_matrix = confusion_matrix(all_true_values, binary_predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    #print("Confusion Matrix:\n", conf_matrix)


    auc_list.append(macro_auc)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    y_true = np.vstack(all_true_values)  # 真实标签
    y_pred = np.vstack(all_predictions)  # 预测值

    df_test = pd.DataFrame(np.hstack((y_true, y_pred)), columns=[
        'True_NR-AR', 'True_NR-AR-LBD', 'True_NR-AhR', 'True_NR-Aromatase',
        'True_NR-ER', 'True_NR-ER-LBD', 'True_NR-PPAR-gamma', 'True_SR-ARE',
        'True_SR-ATAD5', 'True_SR-HSE', 'True_SR-MMP', 'True_SR-p53',
        'Pred_NR-AR', 'Pred_NR-AR-LBD', 'Pred_NR-AhR', 'Pred_NR-Aromatase',
        'Pred_NR-ER', 'Pred_NR-ER-LBD', 'Pred_NR-PPAR-gamma', 'Pred_SR-ARE',
        'Pred_SR-ATAD5', 'Pred_SR-HSE', 'Pred_SR-MMP', 'Pred_SR-p53'
        ])


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

