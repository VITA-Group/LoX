# Lox: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch Code for Paper: "Lox: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning", **Gabriel J. Perin¹, Runjin Chen², Xuxi Chen², Nina S. T. Hirata¹, Zhangyang Wang², and Junyuan Hong²**,
*COLM* 2025. 

¹University of São Paulo  ²University of Texas at Austin  

[paper](https://arxiv.org/abs/2506.15606v1) / [code](https://github.com/VITA-Group/LoX)


**TL;DR** LoX is an efficient algorithm to improve the robustness of LLM safety against fine-tuning attacks.


## Overview:

![overview](fig.png)

Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning—even when the additional training data appears benign.
In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning.
Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM.
Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model’s adaptability to new tasks. For instance, LoX \ leads to 11\% to 54\% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX \ to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations.

## Installation

Run the following command to create the environment:

```
conda env create -f environment.yaml
```

Then, download the required datasets by following the instructions in:

```
data/README.md
```

## Running Experiments

Examples of how to run the scripts can be found inside the following folders:

- `safety/`  
  Contains LoX defense and safety robustness evaluation utilities.

- `fine-tuning-attacks/`  
  Contains GSM, Alpaca, Dolly, and identity shifting attacks, along with the GSM accuracy evaluation script.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{perin2025lox,
  title={LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning}, 
  author={Gabrel J. Perin and Runjin Chen and Xuxi Chen and Nina S. T. Hirata and Zhangyang Wang and Junyuan Hong},
  booktitle={COLM},
  year={2025}
}
```

## Acknowledgement

The work of Z. Wang is in part supported by **Good Systems**, a UT Austin Grand Challenge to develop responsible AI technologies.
G. J. Perin and N. S. T. Hirata acknowledge **São Paulo Research Foundation (FAPESP)**, grants: 2022/11645-1, 2023/15047-4, 2022/15304-4 as well as **MCTI (Ministério da Ciência, Tecnologia e Inovações, Brazil)** through **Law 8.248**, **PPI-Softex - TIC 13 - 01245.010222/2022-44**.

