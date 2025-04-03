# Optspa: Adaptive Sparsity and KV Cache Compression for Efficient Large Multimodal Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Introduction
**Optspa** is an adaptive search algorithm that optimizes sparsity patterns and KV cache compression to enhance the efficiency of Large Multimodal Models (LMMs). Utilizing the Tree-structured Parzen Estimator, our method dynamically adjusts pruning ratios and KV cache quantization bandwidth across different LMM layers, using model performance as the optimization objective. This approach uniquely combines pruning with key-value cache quantization and incorporates a fast pruning technique that eliminates the need for additional fine-tuning or weight adjustments, achieving efficient compression without compromising accuracy.

📄 **Paper**: [Enhancing Large Multimodal Models with Adaptive Sparsity and KV Cache Compression] 
📅 **Proceedings of IEEE International Conference on Multimedia Expo (ICME) 2025**

## Key Features
✨ **Adaptive Sparsity Optimization**  
- Dynamic sparsity pattern selection 
- Layer-wise sparsity for optimal pruning
- Compatible with multimodal models like LLaVA 1.5 and 1.6

🛠 **Work in Progress**  
- Compatiable with Qwen-VL in development

## Installation
```bash
git clone https://github.com/yourusername/optspa.git
cd optspa
```
Please follow the INSTALL.md and the install instructions of LLava

## Usage
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \  
    --prune_method optspa \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```

## Acknowledgements
This project is built upon:
- [Wanda: Pruning by Weights and Activations](https://github.com/locuslab/wanda)
- [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
We thank the authors for their excellent work!


