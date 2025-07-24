## Data Instructions

To prepare the required datasets, please follow the steps below.
### 1. GSM8K Data

Download the following files and place them in `./gsm/`:

- [`train.jsonl`](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/train.jsonl)
- [`test.jsonl`](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)

> Source: [OpenAI Grade School Math Dataset](https://github.com/openai/grade-school-math)

---

### 2. Dolly (No Safety)

Download the Dolly dataset **without safety annotations**:

- [`databricks-dolly-15k-no-safety.jsonl`](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl)

> Place it in this folder.

---

### 3. Alpaca (No Safety)

Download the Alpaca dataset **without safety annotations**:

- [`alpaca_data_no_safety.json`](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/ft_datasets/alpaca_dataset/alpaca_data_no_safety.json)

> Place it in this folder.

---

### 4. AdvBench

Download the adversarial benchmark dataset:

- [`harmful_behaviors.csv`](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)

> Place it in this folder.

---

### 5. ID Attack

Download the in-distribution (ID) attack dataset:

- [`train.json`](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/ft_datasets/aoa_dataset/train.json)

> Rename it to `id_attack.jsonl` and place it in this folder.

---

### 6. SafeInst

Download the Safe Instructions dataset:

- [`safety_only_data_Instructions.json`](https://github.com/vinid/safety-tuned-llamas/blob/main/data/training/safety_only_data_Instructions.json)

> Place it in this folder.

---

### 7. Pure Bad

**Notice:** This dataset contains harmful or unsafe content. For safety and ethical reasons, we have chosen **not to release** this portion of the data.

If you wish to implement a similar type of attack for research purposes, you can sample 100 examples from the [Anthropic Red Teaming dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) and organize them following the structure demonstrated in [this notebook](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/tier1-harmful-examples-demonstration.ipynb).


