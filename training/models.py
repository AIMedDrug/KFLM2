import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
from torch_geometric.nn import GlobalAttention

# 定义稀疏注意力机制
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, k):
        super(SparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k = k
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 定义线性变换层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        k = min(self.k, seq_len) 
        
        # 线性投影
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, num_heads, seq_len, seq_len]
        
        # 在 seq_len 维度上进行 topk 操作
        topk_scores, topk_indices = torch.topk(scores, k, dim=-1)  # 在 seq_len 维度上选择 topk
        
        # 创建稀疏掩码
        sparse_mask = torch.zeros_like(scores)
        sparse_mask.scatter_(-1, topk_indices, 1)
        
        # 应用掩码并计算 softmax
        sparse_scores = scores * sparse_mask
        attention_weights = F.softmax(sparse_scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 拼接多头注意力结果
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 最终线性变换
        output = self.out(output)
        
        return output


# GCN 模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim*2)
        self.fc = nn.Linear(input_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # 全局平均池化
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 处理 SMILES embedding 的模型
class EmbeddingDNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.1):
        super(EmbeddingDNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# 联合模型
class CombinedSparseModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedSparseModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.sparse_attention = SparseAttention(embed_dim=hidden_dim*2, num_heads=4, k=5)  # 稀疏注意力机制
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)

        gcn_output_squeeze = gcn_output.unsqueeze(1)
        # 拼接两个模型的输出
        embedding_output_squeeze = embedding_output.squeeze(1)
        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        combined = torch.cat((embedding_output, gcn_output_squeeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 应用稀疏注意力机制
        combined = self.sparse_attention(combined)
        
        combined = combined.squeeze(1)
        combined = self.fc1(combined)
        combined = self.dropout(combined)
        self.hidden_output = combined.detach()
        
        # 最终输出
        return self.fc_final(combined)
class CombinedModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)
        #torch.Size([32, 1, 64]) torch.Size([32, 64])

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_unsqueeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 保存隐藏层输出
        combined = combined.squeeze(1)
        
        self.hidden_output = combined.detach()

        combined = self.fc1(combined)
        combined = self.dropout(combined)
        # 最终输出
        return self.fc_final(combined)

class CombinedModelClass(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5, num_classes=1):
        super(CombinedModelClass, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, num_classes)  # 最终输出层
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)
        #torch.Size([32, 1, 64]) torch.Size([32, 64])

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_unsqueeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 保存隐藏层输出
        combined = combined.squeeze(1)
        
        self.hidden_output = combined.detach()

        combined = self.fc1(combined)
        combined = self.dropout(combined)
        # 最终输出
        return self.fc_final(combined)
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super(EmbeddingModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim, dropout)
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)
    def forward(self, smiles_embedding):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        return self.fc_final (embedding_output)



class EmbeddingModelClass(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, num_classes=3):
        super(EmbeddingModelClass, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim, dropout)
        self.fc_final = nn.Linear(hidden_dim, num_classes)  # 输出层维度由 num_classes 决定
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, smiles_embedding):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        embedding_output = self.dropout(embedding_output)
        output = self.fc_final(embedding_output)  # 输出 logits
        return output

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.1):
        super(GCNModel, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim=16, dropout=0.1)
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, graph_data):
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        return self.fc_final(gcn_output)

class GCNModelClass(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.1, num_classes=1):
        super(GCNModelClass, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim=16, dropout=0.1)
        self.fc_final = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
    def forward(self, graph_data):
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        gcn_output = self.dropout(gcn_output)
        return self.fc_final(gcn_output)



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by num_heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N = x.shape[0]  # batch size
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        # Split the embedding into multiple heads
        values = self.values(x).view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.keys(x).view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.queries(x).view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nqhk", [queries, keys])  # (N, query_len, key_len, num_heads)
        attention = F.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nqhk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_dim
        )

        return self.fc_out(out)

class CombinedMHAModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedMHAModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.multihead_attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=4)  # 多头注意力机制
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)

        # 拼接两个模型的输出
        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_squeeze), dim=2)  # 拼接后维度为 [batch_size, 128]
    
        # 应用多头注意力机制
        combined = self.multihead_attention(combined)
        
        combined = combined.squeeze(1)
        combined = self.fc1(combined)
        self.hidden_output = combined.detach()
        combined = self.dropout(combined)
        
        # 最终输出
        return self.fc_final(combined)

class CombinedAttModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedAttModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        
        # 注意力机制模块：用于动态分配权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入是两种表示的拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个标量权重
            nn.Sigmoid()  # 使用 Sigmoid 限制权重在 [0, 1]
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)  # 输出维度: [batch_size, 1, hidden_dim]
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)  # 输出维度: [batch_size, hidden_dim]
        
        # 调整维度
        embedding_output = embedding_output.squeeze(1)  # [batch_size, hidden_dim]
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)
        

        # 保存隐藏层输出
        self.hidden_embedding = embedding_output.detach()
        self.hidden_gcn = gcn_output.detach()
        
        # 注意力机制
        combined = torch.cat((embedding_output, gcn_output), dim=1)  # 拼接两种表示
        attention_weight = self.attention(combined)  # 计算注意力权重，维度: [batch_size, 1]
        
        # 动态分配权重
        weighted_embedding = embedding_output * attention_weight  # 加权 embedding_output
        weighted_gcn = gcn_output * (1 - attention_weight)  # 加权 gcn_output
        
        combined_to_fc = weighted_embedding + weighted_gcn  # [batch_size, hidden_dim]
        
        self.hidden_output = combined.detach()
        self.weight = attention_weight.detach() 
        combined_to_fc = self.fc1(combined_to_fc)
        combined_to_fc = self.dropout(combined_to_fc)
        
        return self.fc_final(combined_to_fc)

class CombinedAttModelClass(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5,  num_classes=1):
        super(CombinedAttModelClass, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        
        # 注意力机制模块：用于动态分配权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入是两种表示的拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个标量权重
            nn.Sigmoid()  # 使用 Sigmoid 限制权重在 [0, 1]
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim,  num_classes)  # 最终输出层
        self.dropout = nn.Dropout(dropout)
        self. num_classes =  num_classes

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)  # 输出维度: [batch_size, 1, hidden_dim]
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)  # 输出维度: [batch_size, hidden_dim]
        
        # 调整维度
        embedding_output = embedding_output.squeeze(1)  # [batch_size, hidden_dim]
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)
        

        # 保存隐藏层输出
        self.hidden_embedding = embedding_output.detach()
        self.hidden_gcn = gcn_output.detach()
        
        # 注意力机制
        combined = torch.cat((embedding_output, gcn_output), dim=1)  # 拼接两种表示
        attention_weight = self.attention(combined)  # 计算注意力权重，维度: [batch_size, 1]
        
        # 动态分配权重
        weighted_embedding = embedding_output * attention_weight  # 加权 embedding_output
        weighted_gcn = gcn_output * (1 - attention_weight)  # 加权 gcn_output
        
        combined_to_fc = weighted_embedding + weighted_gcn  # [batch_size, hidden_dim]
        
        self.hidden_output = combined.detach()
        self.weight = attention_weight.detach() 
        combined_to_fc = self.fc1(combined_to_fc)
        combined_to_fc = self.dropout(combined_to_fc)
        
        return self.fc_final(combined_to_fc)

class CombinedModel_test(nn.Module):
    def __init__(self, embedding_dim, graph_dim, hidden_dim=64, dropout=0.5):
        super(CombinedModel_test, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=graph_dim, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)
        #torch.Size([32, 1, 64]) torch.Size([32, 64])

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_squeeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 保存隐藏层输出
        combined = combined.squeeze(1)
        
        combined = self.fc1(combined)
        self.hidden_output = combined.detach()
        combined = self.dropout(combined)
        # 最终输出
        return self.fc_final(combined)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim*2)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.1):
        super(GATModel, self).__init__()
        self.gat = GAT(input_dim, hidden_dim=16, dropout=0.1)
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, graph_data):
        # 处理图数据
        gat_output = self.gat(graph_data)
        gat_output = self.dropout(gat_output)
        return self.fc_final(gat_output)

class CombinedGATModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedGATModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GAT(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)
        #torch.Size([32, 1, 64]) torch.Size([32, 64])

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_unsqueeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 保存隐藏层输出
        combined = combined.squeeze(1)
        
        self.hidden_output = combined.detach()

        combined = self.fc1(combined)
        combined = self.dropout(combined)
        # 最终输出
        return self.fc_final(combined)

class CombinedGATModel_test(nn.Module):
    def __init__(self, embedding_dim, graph_dim, hidden_dim=64, dropout=0.5):
        super(CombinedGATModel_test, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GAT(input_dim=graph_dim, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)  # 最终输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)
        #print(embedding_output.shape, gcn_output.shape)
        #torch.Size([32, 1, 64]) torch.Size([32, 64])

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 拼接两个模型的输出
        combined = torch.cat((embedding_output, gcn_output_squeeze), dim=2)  # 拼接后维度为 [batch_size, 128]
        #print(combined.shape)
    
        # 保存隐藏层输出
        combined = combined.squeeze(1)
        
        combined = self.fc1(combined)
        self.hidden_output = combined.detach()
        combined = self.dropout(combined)
        # 最终输出
        return self.fc_final(combined)


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.fc_query = nn.Linear(input_dim, input_dim)
        self.fc_key = nn.Linear(input_dim, input_dim)
        self.fc_value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)

        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权输出
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

class CombinedSepAttModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedSepAttModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.attention_smiles = AttentionLayer(hidden_dim)
        self.attention_graph = AttentionLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 拼接后输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)

        # 将输出进行 squeeze 操作
        #embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)  # 添加维度以匹配注意力输入

        # 对 SMILES 嵌入应用注意力机制
        attended_smiles, _ = self.attention_smiles(embedding_output)  # 添加维度以匹配
        attended_smiles = attended_smiles.squeeze(1)

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 对图数据应用注意力机制
        attended_graph, _ = self.attention_graph(gcn_output_squeeze)

        # 将两个模态的加权输出进行拼接
        combined = torch.cat((attended_smiles, attended_graph.squeeze(1)), dim=1)  # 拼接后维度为 [batch_size, hidden_dim * 2]

        self.hidden_output = combined.detach()

        combined = self.fc1(combined)
        combined = self.dropout(combined)

        # 最终输出
        return self.fc_final(combined)

class CombinedSepAttModelClass(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5, num_classes=1):
        super(CombinedSepAttModelClass, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.attention_smiles = AttentionLayer(hidden_dim)
        self.attention_graph = AttentionLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 拼接后输出层
        self.fc_final = nn.Linear(hidden_dim, num_classes)  # 最终输出层
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)
        
        # 处理图数据
        gcn_output = self.gcn(graph_data)

        # 将输出进行 squeeze 操作
        #embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)  # 添加维度以匹配注意力输入

        # 对 SMILES 嵌入应用注意力机制
        attended_smiles, _ = self.attention_smiles(embedding_output)  # 添加维度以匹配
        attended_smiles = attended_smiles.squeeze(1)

        embedding_output_squeeze = embedding_output.squeeze(1)
        gcn_output_squeeze = gcn_output.unsqueeze(1)

        self.hidden_embedding = embedding_output_squeeze.detach()
        self.hidden_gcn = gcn_output.detach()
        # 对图数据应用注意力机制
        attended_graph, _ = self.attention_graph(gcn_output_squeeze)

        # 将两个模态的加权输出进行拼接
        combined = torch.cat((attended_smiles, attended_graph.squeeze(1)), dim=1)  # 拼接后维度为 [batch_size, hidden_dim * 2]

        self.hidden_output = combined.detach()

        combined = self.fc1(combined)
        combined = self.dropout(combined)

        # 最终输出
        return self.fc_final(combined)

class AttentionLayer2(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer2, self).__init__()
        self.fc_query = nn.Linear(input_dim, input_dim)
        self.fc_key = nn.Linear(input_dim, input_dim)
        self.fc_value = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        query = self.fc_query(query)
        key = self.fc_key(key)
        value = self.fc_value(value)

        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权输出
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

class CombinedCrossAttModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super(CombinedCrossAttModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)  # 假设输入维度为 64
        self.attention1 = AttentionLayer2(hidden_dim)  # 用于 SMILES 和图的交叉注意力
        self.attention2 = AttentionLayer2(hidden_dim)  # 用于图和 SMILES 的交叉注意力
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 拼接后输出层
        self.fc_final = nn.Linear(hidden_dim, 1)  # 最终输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        # 处理 SMILES embedding
        embedding_output = self.embedding_model(smiles_embedding)

        # 处理图数据
        gcn_output = self.gcn(graph_data)

        gcn_output_unsqueeze = gcn_output.unsqueeze(1)  # 添加维度以匹配注意力输入
        # 对 SMILES 嵌入和图数据进行交叉注意力计算
        attended_smiles, attention_weights_smiles = self.attention1(embedding_output, gcn_output_unsqueeze, gcn_output_unsqueeze)
        attended_graph, attention_weights_graph = self.attention2(gcn_output_unsqueeze, embedding_output, embedding_output)

        # 拼接两个模态的加权输出
        combined = torch.cat((attended_smiles.squeeze(1), attended_graph.squeeze(1)), dim=1)  # 拼接后维度为 [batch_size, hidden_dim * 2]

        combined = self.fc1(combined)
        combined = self.dropout(combined)

        # 最终输出
        return self.fc_final(combined)

class CombinedCrossAttClassModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, num_classes=3, dropout=0.5):  # num_classes 为类别数
        super(CombinedCrossAttClassModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)
        self.attention1 = AttentionLayer2(hidden_dim)
        self.attention2 = AttentionLayer2(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, num_classes)  # 输出层为 num_classes，用于多分类
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_embedding, graph_data):
        embedding_output = self.embedding_model(smiles_embedding)
        gcn_output = self.gcn(graph_data)
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)

        attended_smiles, attention_weights_smiles = self.attention1(embedding_output, gcn_output_unsqueeze, gcn_output_unsqueeze)
        attended_graph, attention_weights_graph = self.attention2(gcn_output_unsqueeze, embedding_output, embedding_output)

        combined = torch.cat((attended_smiles.squeeze(1), attended_graph.squeeze(1)), dim=1)
        combined = self.fc1(combined)
        combined = self.dropout(combined)

        # Softmax 激活，用于多分类
        return F.softmax(self.fc_final(combined), dim=1)


class CombinedCrossAttBiClassModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5, num_classes=1):
        super(CombinedCrossAttBiClassModel, self).__init__()
        self.embedding_model = EmbeddingDNN(embedding_dim, hidden_dim)
        self.gcn = GCN(input_dim=64, hidden_dim=hidden_dim)
        self.attention1 = AttentionLayer2(hidden_dim)
        self.attention2 = AttentionLayer2(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, num_classes)  # 输出层为num_classes，用于多分类或二分类
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, smiles_embedding, graph_data, activation_fn=None):
        embedding_output = self.embedding_model(smiles_embedding)
        gcn_output = self.gcn(graph_data)
        gcn_output_unsqueeze = gcn_output.unsqueeze(1)

        attended_smiles, attention_weights_smiles = self.attention1(embedding_output, gcn_output_unsqueeze, gcn_output_unsqueeze)
        attended_graph, attention_weights_graph = self.attention2(gcn_output_unsqueeze, embedding_output, embedding_output)

        combined = torch.cat((attended_smiles.squeeze(1), attended_graph.squeeze(1)), dim=1)
        combined = self.fc1(combined)
        combined = self.dropout(combined)

        output = self.fc_final(combined)


        return output

