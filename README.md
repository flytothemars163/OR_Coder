# OR-Coder: Optimization Problem Code Generation with Fine-tuned LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

OR-Coder is a specialized code generation framework that fine-tunes large language models to solve complex operations research and optimization problems. This project demonstrates how to effectively fine-tune Qwen2.5-7B-Instruct for generating optimization code from natural language descriptions.

## 🎯 Project Overview

### Background
Large language models have shown remarkable capabilities in code generation, but they often struggle with complex mathematical optimization problems that require precise mathematical modeling and solver integration. OR-Coder addresses this gap by providing specialized fine-tuning techniques for optimization problem solving.

### Problem Statement
- **Data Construction**: Creating high-quality training data for optimization problems
- **Fine-tuning Methodology**: Selecting and implementing effective fine-tuning strategies
- **Performance Evaluation**: Developing robust evaluation metrics for optimization code generation

## 📊 Experimental Setup

### Datasets

#### 1. LORA Fine-tuning Data: IndustryOR
- **Description**: Industrial optimization problems with real-world scenarios
- **Size**: Comprehensive dataset covering various industry domains
- **Content**: Natural language problem descriptions + mathematical models + solver code

#### 2. PPO Training Data: OR-Instruct-Data-3K  
- **Description**: 3,000 diverse optimization instruction-response pairs
- **Variety**: Linear programming, integer programming, mixed-integer programming, etc.
- **Quality**: Expert-verified mathematical models and code solutions

#### 3. Evaluation Data: BWOR
- **Description**: Business World Optimization Problems benchmark
- **Purpose**: Standardized evaluation of optimization code generation capabilities
- **Metrics**: Pass@1 accuracy, code correctness, solution optimality

### Model Architecture
- **Base Model**: Qwen2.5-7B-Instruct
- **Fine-tuning Methods**: 
  - LoRA (Low-Rank Adaptation)
  - PPO (Proximal Policy Optimization)
- **Training Framework**: Hugging Face Transformers + TRL

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/OR-Coder.git
cd OR-Coder

# Create conda environment
conda create -n or-coder python=3.10
conda activate or-coder

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Convert IndustryOR dataset for SFT
python data/convert_industryor_to_sft_jsonl.py

# Convert OR-Instruct dataset for training
python data/convert_or_instruct_to_sft_jsonl.py
```

### Training

#### Supervised Fine-tuning (SFT)
```bash
cd train
./start_sft.sh
```

#### Proximal Policy Optimization (PPO)
```bash
cd train  
./start_ppo.sh
```

### Evaluation
```bash
python evaluation/or_llm_eval.py --model your-model-path --dataset BWOR
```

## 📈 Results

### Performance Comparison

| Model | Pass@1 Rate | Improvement |
|-------|------------|-------------|
| Baseline (Qwen2.5-7B-Instruct) | 9.7% | - |
| + LoRA Fine-tuning | 14.6% | +50.5% |
| + PPO Training | 11.0% | +13.4% |

### Key Findings

1. **LoRA Effectiveness**: LoRA fine-tuning provides the most significant improvement (+50.5%) in optimization code generation accuracy.

2. **Specialization Benefits**: Domain-specific fine-tuning dramatically improves performance on optimization problems compared to general-purpose code generation.

3. **Data Quality Impact**: High-quality, domain-specific training data is crucial for effective fine-tuning.

## 🏗️ Project Structure

```
OR-Coder/
├── 📁 data/                         # Data processing and conversion scripts
│   ├── convert_industryor_to_sft_jsonl.py  # Convert IndustryOR to SFT format
│   └── convert_or_instruct_to_sft_jsonl.py  # Convert OR-Instruct to training format
│
├── 📁 evaluation/                    # Model evaluation framework
│   └── or_llm_eval.py               # Main evaluation script for optimization problems
│
├── 📁 train/                        # Training scripts and configurations
│   ├── start_sft.sh                 # Shell script to launch SFT training
│   ├── start_ppo.sh                 # Shell script to launch PPO training
│   ├── train_sft.py                 # Supervised Fine-tuning implementation
│   └── train_ppo_lora.py            # PPO with LoRA fine-tuning implementation
│
├── 📄 requirement.txt               # Python dependencies and package requirements
├── 📄 README.md                     # Project documentation (this file)
└── 📄 LICENSE                       # MIT License file
```

### Detailed File Descriptions

#### Data Processing Scripts
- **`convert_industryor_to_sft_jsonl.py`**: Converts the IndustryOR dataset into JSONL format suitable for supervised fine-tuning, handling natural language to mathematical model transformations.

- **`convert_or_instruct_to_sft_jsonl.py`**: Processes the OR-Instruct-Data-3K dataset, creating instruction-response pairs for training optimization code generation models.

#### Training Scripts
- **`train_sft.py`**: Implements supervised fine-tuning with LoRA adaptation, supporting Qwen2.5-7B-Instruct and other transformer models.

- **`train_ppo_lora.py`**: Combines Proximal Policy Optimization with LoRA fine-tuning for reinforcement learning based training.

- **`start_sft.sh`**: Bash script that sets up environment variables and launches SFT training with optimal hyperparameters.

- **`start_ppo.sh`**: Automation script for PPO training with configured reward functions and training parameters.

#### Evaluation Framework
- **`or_llm_eval.py`**: Comprehensive evaluation script that tests model performance on BWOR benchmark, measuring Pass@1 rates and code correctness metrics.

### Configuration Details

#### Training Hyperparameters
```yaml
# SFT Configuration
learning_rate: 2e-4
batch_size: 16
lora_rank: 16
lora_alpha: 32
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# PPO Configuration  
kl_penalty: 0.2
clip_range: 0.2
gae_lambda: 0.95
value_loss_coef: 0.1
```

#### Dataset Specifications
```yaml
IndustryOR:
  size: Comprehensive industrial optimization problems
  format: Natural language + mathematical models + solver code
  domains: Manufacturing, logistics, supply chain, resource allocation

OR-Instruct-Data-3K:
  size: 3,000 instruction-response pairs  
  variety: LP, MIP, NLP, combinatorial optimization
  quality: Expert-verified solutions

BWOR:
  purpose: Evaluation benchmark
  metrics: Pass@1, code correctness, solution optimality
  difficulty: Business-world complexity level
```