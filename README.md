# KFLM2: Knowledge-Fused Large Language Model for dual-Modal learning for molecular property prediction

## Introduction

We propose a multi-modality learning framework for molecular property prediction by integrating molecular SMILES and molecular graph. The aim is to utilize the capabilities of advanced large language models (LLMs), strengthened with specialized knowledge in the field of drug discovery. We identified DeepSeek-R1-Distill-Qwen-1.5B as the  base model from three DeepSeek-R1 distilled LLMs. By leveraging dual-modality learning, KFLM2 adapts to domain-specific datasets, enhancing prediction accuracy. The model operates in two stages: (1) initial fine-tuning of the LLMs and (2) dataset-oriented training to optimize task-specific performance.

 We recommend installation via conda. You can create a conda environment from the environment.yml file.

```shell
conda env create -f conda_environment.yml
```

## Directories

| # | task | Directory  | 
|:-----|:-----|-----:|
|1.  | **dataset** | [`dataset`]|
|2.  | **fine-tuning** | [`fine_tuning`]|
|3.  | **hyperparameter** | [`hyperparameter`]|
|4.  | **training** | [`task_training`]|
|5.  | **prompt** | [`prompt`]|



## Fine-tuning

Please fine-tune the LLMs using the ZINC or ChEMBL dataset. In current study, we select the ZINC dataset to fine-tune the DeepSeek Distilled 1.5B, 7B, 8B and ChemDFM. The names of the distilled model checkpoints are: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (abbreviated as DS_1.5B), deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (abbreviated as DS_7B), deepseek-ai/DeepSeek-R1-Distill-Llama-8B (abbreviated as DS_8B), and OpenDFM/ChemDFM-v1.5-8B)

To avoid large GPU memory dependency, which can often lead to performance bottlenecks and limitations in the training and inference processes, we conduct a comprehensive analysis of various available models. After carefully evaluating their memory requirements, computational complexity, and compatibility with our system setup, we ultimately select DS_1.5B for our following base model. DS_1.5B has demonstrated relatively lower GPU memory consumption while still maintaining satisfactory performance in relevant tasks, making it a suitable choice to serve as the foundation for our subsequent research and development work. This selection aims to ensure smooth operations and efficient utilization of resources during the model training and deployment phase

The fine-tuning can be completed as the follows:

```
python DeepkSeek_R1_1.5B_Finetuning_LoRA.py
```

## Dataset-oriented training

Performance testing datasets include five regression datasets and four classification 
datasets. All datasets were collected from MoleculeNet, except for the Llinas set, 
which was obtained from a recent study, and pKa dataset, which was extracted 
from DataWarrior software. All datasets are partitioned into training, validation, 
and test sets in a ratio of 8:1:1, except for Llinas, whose tests have already been 
determined.

### Code Structure
Our implementation consists of modular components:

Molecular feature processing:

featerize_smiles.py handles conversion of molecules into both LLM-based embeddings and graph representations

Model architecture:

model.py contains core model definitions, including two distinct attention mechanisms.

### Task-Specific Implementation: take SAMPL as the example.
For the SAMPL dataset, we provide the following training pipelines, which can run directly after installation:

**Embedding-based DNN training**: train_combined_base_SAMPL.py

**Graph-based GCN learning**: train_gcn_indepent_SAMPL.py

**Hybrid approach combining both embeddings and graph features**: train_combined_SAMPL.py

**Weighted concatenation**: train_combinedAtt_SAMPL.py

**Attention concatenation**: train_combinedSepAtt_SAMPL.py



## LLM Prompting Evaluation

To further assess model performance, we conduct prompt-based evaluations using large language models (LLMs). Specifically, we fine-tune the model on a small-scale dataset, then leverage LLMs to generate outputs based on structured prompts. The results are systematically analyzed to evaluate both robustness and generalization capabilities.

```bash
python fine_tune_prompt_unsloth_regression.py
```

## Todo
- [x] Clean the codes and remove unuse function
- [x] Implemente MoE
- [x] Include 3D information in molecular graph

