# Lox: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

**Gabriel J. Perin¹, Runjin Chen², Xuxi Chen², Nina S. T. Hirata¹, Zhangyang Wang², Junyuan Hong²**

¹University of São Paulo  ²University of Texas at Austin  

https://arxiv.org/abs/2506.15606v1

## Abstract:

Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning—even when the additional training data appears benign.
In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning.
Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM.
Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model’s adaptability to new tasks. For instance, LoX \ leads to 11\% to 54\% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX \ to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations.
