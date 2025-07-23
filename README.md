# Lox: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

**Gabriel J. Perin¹, Runjin Chen², Xuxi Chen², Nina S. T. Hirata¹, Zhangyang Wang², Junyuan Hong²**

¹University of São Paulo  ²University of Texas at Austin  

https://arxiv.org/abs/2506.15606v1

## Overview:

![Figure description](fig.png)

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

---

## Running the Script

Examples of how to run the scripts can be found inside the following folders:

- `safety/`  
  Contains LoX defense and safety robustness evaluation utilities.

- `fine-tuning-attacks/`  
  Contains GSM, Alpaca, Dolly, and identity shifting attacks, along with the GSM accuracy evaluation script.

## Citation

If you use this work, please cite:

```bibtex
@misc{perin2025loxlowrankextrapolationrobustifies,
      title={LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning}, 
      author={Gabrel J. Perin and Runjin Chen and Xuxi Chen and Nina S. T. Hirata and Zhangyang Wang and Junyuan Hong},
      year={2025},
      eprint={2506.15606},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.15606}, 
}
```

## Acknowledgement

The work of Z. Wang is in part supported by **Good Systems**, a UT Austin Grand Challenge to develop responsible AI technologies.

G. J. Perin and N. S. T. Hirata acknowledge **São Paulo Research Foundation (FAPESP)**, grants:

- 2022/11645-1  
- 2023/15047-4  
- 2022/15304-4  

as well as **MCTI (Ministério da Ciência, Tecnologia e Inovações, Brazil)** through **Law 8.248**, **PPI-Softex - TIC 13 - 01245.010222/2022-44**.


