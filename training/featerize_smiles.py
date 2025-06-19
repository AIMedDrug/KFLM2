import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from rdkit import Chem
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import json
import torch.nn as nn

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, explicit_H=True):
    # feature dim 64
    # atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_symbols = [
        'C','N', 'O','S','F','Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As', 'Al','I','B','V', 'K','Tl','Yb','Sb',
        'Sn','Ag', 'Pd','Co', 'Se', 'Ti','Zn','H', 'Li', 'Ge', 'Cu', 'Au','Ni','Cd','In','Mn', 'Zr','Cr','Pt','Hg', 'Pb', 'Unknown'
        #'C','N', 'O','S','F','P','Cl','Br','Mg', 'Ca','Fe','As', 'Al','H', 'Zn', 'Pt', 'Unknown',
      ]
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def atom_features28(mol, graph, explicit_H=True):
    # feature dim 28
    atom_symbols = ['P', 'N', 'C', 'Cl', 'Br', 'I', 'S', 'O', 'F', ]
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(), [1, 2, 3, 4, 5]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                ]) + [atom.GetIsAromatic()]
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    edge_set = set()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if (i, j) not in edge_set and (j, i) not in edge_set:  # 检查边是否已经添加
            graph.add_edge(i, j)
            edge_set.add((i, j))

def get_edgefeat_index(mol, graph):
    edge_set = set()
    edge_features = []  # 用于保存边特征
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if (i, j) not in edge_set and (j, i) not in edge_set:
            graph.add_edge(i, j)
            edge_set.add((i, j))
            
            # 处理键特征
            bond_type = bond.GetBondTypeAsDouble()  # 获取键类型的数值表示
            edge_features.append(bond_type)  # 将键类型添加到边特征列表中
            
    # 将边特征转换为张量
    if edge_features:
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_features_tensor = torch.empty((0,), dtype=torch.float32)  # 空边特征
    
    return edge_features_tensor

def mol2graphfeat(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_index = get_edgefeat_index(mol, graph)
    
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    x = x[:, :64]  # 选择前64维特征

    edges = list(graph.edges(data=False))
    if not edges:
        return x, torch.empty((2, 0), dtype=torch.long)  # 返回空的 edge_index

    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in edges]).T
    return x, edge_index

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    x = x[:, :64]
    #print(graph.edges(data=False))
    edges = list(graph.edges(data=False))
    if not edges:
        # 处理空边的情况，返回默认值
        return x, torch.empty((2, 0), dtype=torch.long)  # 返回空的 edge_index

    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in edges]).T
    #edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    return x, edge_index

class SmilesDataset(Dataset):
    def __init__(self, smiles, logP, model, tokenizer):
        self.smiles = smiles
        self.logP = logP
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        logP = self.logP[idx]

        # 生成分子图
        graph_data = self.smiles_to_graph(smiles)

        if graph_data is not None:
            # 只有通过的 SMILES 才生成 embedding
            embedding = self.generate_embedding(smiles)
            return (torch.tensor(embedding, dtype=torch.float32), graph_data), torch.tensor(logP, dtype=torch.float32)
        else:
            # 如果 SMILES 无法解析，返回 None
            return None, torch.tensor(logP, dtype=torch.float32)

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.cpu().numpy()

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() <= 2:
            return None

        x, edge_index = mol2graph(mol)
        return Data(x=x, edge_index=edge_index)

class SmilesDatasetfeat(Dataset):
    def __init__(self, smiles, logP, model, tokenizer):
        self.smiles = smiles
        self.logP = logP
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        logP = self.logP[idx]

        # 生成分子图
        graph_data = self.smiles_to_graph(smiles)

        if graph_data is not None:
            # 只有通过的 SMILES 才生成 embedding
            embedding = self.generate_embedding(smiles)
            return (torch.tensor(embedding, dtype=torch.float32), graph_data), torch.tensor(logP, dtype=torch.float32)
        else:
            # 如果 SMILES 无法解析，返回 None
            return None, torch.tensor(logP, dtype=torch.float32)

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.cpu().numpy()

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() <= 2:
            return None

        x, edge_index = mol2graphfeat(mol)
        return Data(x=x, edge_index=edge_index)

def mol2graph28(mol):
    graph = nx.Graph()
    atom_features28(mol, graph)
    get_edge_index(mol, graph)
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    x = x[:, :28]
    #print(graph.edges(data=False))
    edges = list(graph.edges(data=False))
    if not edges:
        # 处理空边的情况，返回默认值
        return x, torch.empty((2, 0), dtype=torch.long)  # 返回空的 edge_index

    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in edges]).T
    #edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    return x, edge_index

class SmilesDataset28(Dataset):
    def __init__(self, smiles, logP, model, tokenizer):
        self.smiles = smiles
        self.logP = logP
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        logP = self.logP[idx]

        # 生成分子图
        graph_data = self.smiles_to_graph(smiles)

        if graph_data is not None:
            # 只有通过的 SMILES 才生成 embedding
            embedding = self.generate_embedding(smiles)
            return (torch.tensor(embedding, dtype=torch.float32), graph_data), torch.tensor(logP, dtype=torch.float32)
        else:
            # 如果 SMILES 无法解析，返回 None
            return None, torch.tensor(logP, dtype=torch.float32)

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.cpu().numpy()

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() <= 2:
            return None

        x, edge_index = mol2graph28(mol)
        return Data(x=x, edge_index=edge_index)


class SMILESToken:
    def __init__(self, vocab_file, embedding_dim=128):
        self.vocab = self.load_vocab(vocab_file)
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_dim)

    def load_vocab(self, file_path):
        """从 JSON 文件加载词汇表"""
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def encode_smiles(self, smiles):
        """将 SMILES 字符串编码为对应的索引列表"""
        return [self.vocab.get(char, self.vocab['UNK']) for char in smiles]

    def encode_batch(self, smiles_list):
        """将一批 SMILES 字符串编码为索引张量"""
        encoded = [self.encode_smiles(smiles) for smiles in smiles_list]
        #max_length = max(len(seq) for seq in encoded)
        max_length = 80
        padded_encoded = []
        for seq in encoded:
            if len(seq) < max_length:
                padded_encoded.append(seq + [self.vocab['PAD']] * (max_length - len(seq)))
            else:
                padded_encoded.append(seq[:max_length])

        return torch.tensor(padded_encoded, dtype=torch.float)

    def get_embeddings(self, smiles_list):
        """获取 SMILES 的嵌入表示"""
        encoded_tensor = self.encode_batch(smiles_list)
        return self.embedding(encoded_tensor)

class SMILESTokenDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.smiles = self.data['smiles'].tolist()
        self.logP = self.data['expt'].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        logP_value = self.logP[idx]
        # 直接返回 SMILES 和 logP 值
        return smiles, logP_value

